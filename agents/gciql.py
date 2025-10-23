import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue


class GCIQLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit Q-learning (GCIQL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        q1, q2 = self.network.select("target_critic")(
            batch["observations"], batch["value_goals"], batch["actions"]
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config["expectile"]).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def critic_loss(self, batch, grad_params):
        next_v = self.network.select("value")(
            batch["next_observations"], batch["value_goals"]
        )
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v

        q1, q2 = self.network.select("critic")(
            batch["observations"],
            batch["value_goals"],
            batch["actions"],
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        if self.config["actor_loss"] == "awr":
            v = self.network.select("value")(
                batch["observations"], batch["actor_goals"]
            )
            q1, q2 = self.network.select("critic")(
                batch["observations"], batch["actor_goals"], batch["actions"]
            )
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config["alpha"])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], params=grad_params
            )
            log_prob = dist.log_prob(batch["actions"])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                "actor_loss": actor_loss,
                "adv": adv.mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }

            return actor_loss, actor_info
        elif self.config["actor_loss"] == "ddpgbc":
            dist = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], params=grad_params
            )
            if self.config["const_std"]:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select("critic")(
                batch["observations"], batch["actor_goals"], q_actions
            )
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch["actions"])

            bc_loss = -(self.config["alpha"] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
                "q_mean": q.mean(),
                "q_abs_mean": jnp.abs(q).mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        dist = self.network.select("actor")(
            observations, goals, temperature=temperature
        )
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch["observations"]
        ex_actions = example_batch["actions"]
        ex_goals = example_batch["actor_goals"]
        action_dim = ex_actions.shape[-1]

        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=1,
        )
        critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
        )

        actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
            state_dependent_std=False,
            const_std=config["const_std"],
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_goals, ex_actions),
            ),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_critic"] = params["modules_critic"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="gciql",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024,) * 4,
            value_hidden_dims=(1024,) * 4,
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            expectile=0.9,
            actor_loss="ddpgbc",  # awr or ddpgbc
            alpha=1.0,
            const_std=True,
            dataset=mlc.ConfigDict(
                dict(
                    dataset_class="GCDataset",
                    value_p_curgoal=0.2,
                    value_p_trajgoal=0.5,
                    value_p_randomgoal=0.3,
                    value_geom_sample=False,
                    actor_p_curgoal=0.0,
                    actor_p_trajgoal=0.5,
                    actor_p_randomgoal=0.5,
                    actor_geom_sample=True,
                    gc_negative=True,
                )
            ),
        )
    )
    return config
