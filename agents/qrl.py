from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import numpy as np
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    MLP,
    GCActor,
    GCDiscreteActor,
    GCIQEValue,
    GCMRNValue,
    LogParam,
    GCValue,
)


class QRLAgent(flax.struct.PyTreeNode):
    """Quasimetric RL (QRL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def value_loss(self, batch, grad_params):
        value_goal_key = (
            "value_goal_observations"
            if self.config["oracle_distill"]
            else "value_goals"
        )
        d_neg, ob_rep, _ = self.network.select("value")(
            batch["observations"], batch[value_goal_key], params=grad_params, info=True
        )
        d_pos = self.network.select("value")(
            batch["observations"], batch["next_observations"], params=grad_params
        )
        lam = self.network.select("lam")(params=grad_params)

        # Apply loss shaping following the original implementation.
        d_neg_loss = (100 * jax.nn.softplus(5 - d_neg / 100)).mean()
        d_pos_loss = (jax.nn.relu(d_pos - 1) ** 2).mean()

        value_loss = d_neg_loss + d_pos_loss * jax.lax.stop_gradient(lam)
        lam_loss = lam * (self.config["eps"] - jax.lax.stop_gradient(d_pos_loss))

        total_loss = value_loss + lam_loss

        if self.config["oracle_distill"]:
            distill_value = self.network.select("oracle_value")(
                jax.lax.stop_gradient(ob_rep),
                goals=batch["value_goals"],
                params=grad_params,
            )
            distill_loss = jnp.square(
                distill_value - jax.lax.stop_gradient(d_neg)
            ).mean()

            total_loss = total_loss + distill_loss

        return total_loss, {
            "total_loss": total_loss,
            "value_loss": value_loss,
            "lam_loss": lam_loss,
            "d_neg_loss": d_neg_loss,
            "d_neg_mean": d_neg.mean(),
            "d_neg_max": d_neg.max(),
            "d_neg_min": d_neg.min(),
            "d_pos_loss": d_pos_loss,
            "d_pos_mean": d_pos.mean(),
            "d_pos_max": d_pos.max(),
            "d_pos_min": d_pos.min(),
            "lam": lam,
        }

    def dynamics_loss(self, batch, grad_params):
        _, ob_reps, next_ob_reps = self.network.select("value")(
            batch["observations"],
            batch["next_observations"],
            info=True,
            params=grad_params,
        )
        # Dynamics model predicts the delta of the next observation.
        pred_next_ob_reps = ob_reps + self.network.select("dynamics")(
            jnp.concatenate([ob_reps, batch["actions"]], axis=-1), params=grad_params
        )

        dist1 = self.network.select("value")(
            next_ob_reps, pred_next_ob_reps, is_phi=True, params=grad_params
        )
        dist2 = self.network.select("value")(
            pred_next_ob_reps, next_ob_reps, is_phi=True, params=grad_params
        )
        dynamics_loss = (dist1 + dist2).mean() / 2

        return dynamics_loss, {
            "dynamics_loss": dynamics_loss,
        }

    def actor_loss(self, batch, grad_params, rng=None):
        assert not self.config["discrete"]

        dist = self.network.select("actor")(
            batch["observations"], batch["actor_goals"], params=grad_params
        )
        if self.config["const_std"]:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

        if self.config["oracle_distill"]:
            _, ob_reps, _ = self.network.select("value")(
                batch["observations"], batch["actor_goal_observations"], info=True
            )
            pred_next_ob_reps = ob_reps + self.network.select("dynamics")(
                jnp.concatenate([ob_reps, q_actions], axis=-1)
            )
            q = -self.network.select("oracle_value")(
                pred_next_ob_reps, batch["actor_goals"]
            )
        else:
            _, ob_reps, goal_reps = self.network.select("value")(
                batch["observations"], batch["actor_goals"], info=True
            )
            pred_next_ob_reps = ob_reps + self.network.select("dynamics")(
                jnp.concatenate([ob_reps, q_actions], axis=-1)
            )
            q = -self.network.select("value")(pred_next_ob_reps, goal_reps, is_phi=True)

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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        dynamics_loss, dynamics_info = self.dynamics_loss(batch, grad_params)
        for k, v in dynamics_info.items():
            info[f"dynamics/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = value_loss + dynamics_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

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
        if not self.config["discrete"]:
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
        ex_latents = np.zeros(
            (ex_observations.shape[0], config["latent_dim"]), dtype=np.float32
        )
        action_dim = ex_actions.shape[-1]

        assert not config["discrete"]

        if config["quasimetric_type"] == "mrn":
            value_def = GCMRNValue(
                hidden_dims=config["value_hidden_dims"],
                latent_dim=config["latent_dim"],
                layer_norm=config["layer_norm"],
            )
        elif config["quasimetric_type"] == "iqe":
            value_def = GCIQEValue(
                hidden_dims=config["value_hidden_dims"],
                latent_dim=config["latent_dim"],
                dim_per_component=8,
                layer_norm=config["layer_norm"],
            )
        else:
            raise ValueError(
                f'Unsupported quasimetric type: {config["quasimetric_type"]}'
            )

        dynamics_def = MLP(
            hidden_dims=(*config["value_hidden_dims"], config["latent_dim"]),
            layer_norm=config["layer_norm"],
        )

        if config["discrete"]:
            actor_def = GCDiscreteActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config["const_std"],
            )

        oracle_value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
        )

        # define the dual lambda variable
        lam_def = LogParam()

        ex_value_goals = ex_observations if config["oracle_distill"] else ex_goals
        network_info = dict(
            value=(value_def, (ex_observations, ex_value_goals)),
            oracle_value=(oracle_value_def, (ex_latents, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
            dynamics=(dynamics_def, np.concatenate([ex_latents, ex_actions], axis=-1)),
            lam=(lam_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="qrl",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024,) * 4,
            value_hidden_dims=(1024,) * 4,
            quasimetric_type="iqe",  # iqe (interval quasimetric embedding) or mrn (metric residual network)
            latent_dim=1024,
            layer_norm=True,
            discount=0.999,
            eps=0.05,
            alpha=0.003,
            const_std=True,
            discrete=False,
            oracle_distill=False,
            dataset=mlc.ConfigDict(
                dict(
                    dataset_class="GCDataset",
                    value_p_curgoal=0.0,
                    value_p_trajgoal=0.0,
                    value_p_randomgoal=1.0,
                    value_geom_sample=True,
                    actor_p_curgoal=0.0,
                    actor_p_trajgoal=0.5,
                    actor_p_randomgoal=0.5,
                    actor_geom_sample=False,
                )
            ),
        )
    )
    return config
