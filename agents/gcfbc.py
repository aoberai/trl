from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField


class GCFBCAgent(flax.struct.PyTreeNode):
    """Goal-conditioned flow behavioral cloning (GCFBC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params, rng=None):
        batch_size, action_dim = batch["actions"].shape
        x_rng, t_rng = jax.random.split(rng, 2)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch["actions"]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        y = x_1 - x_0

        pred = self.network.select("actor_flow")(
            batch["observations"], batch["actor_goals"], x_t, t, params=grad_params
        )

        actor_loss = jnp.mean((pred - y) ** 2)

        actor_info = {
            "actor_loss": actor_loss,
        }

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = actor_loss
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
        actions = jax.random.normal(
            seed,
            (
                *observations.shape[:-1],
                self.config["action_dim"],
            ),
        )
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            vels = self.network.select("actor_flow")(observations, goals, actions, t)
            actions = actions + vels / self.config["flow_steps"]
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
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]

        actor_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
        )

        network_info = dict(
            actor_flow=(
                actor_flow_def,
                (ex_observations, ex_goals, ex_actions, ex_times),
            ),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="gcfbc",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024,) * 4,
            layer_norm=True,
            discount=0.999,
            action_dim=mlc.config_dict.placeholder(int),
            flow_steps=10,
            dataset=mlc.ConfigDict(
                dict(
                    dataset_class="GCDataset",
                    actor_p_curgoal=0.0,
                    actor_p_trajgoal=1.0,
                    actor_p_randomgoal=0.0,
                    actor_geom_sample=True,
                )
            ),
        )
    )
    return config
