import dataclasses
from typing import Any

import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict
from ml_collections import ConfigDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if "valids" in self._dict:
            (self.valid_idxs,) = np.nonzero(self["valids"] > 0)

        (self.terminal_locs,) = np.nonzero(self["terminals"] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if "valids" in self._dict:
            return self.valid_idxs[
                np.random.randint(len(self.valid_idxs), size=num_idxs)
            ]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if "next_observations" not in result:
            result["next_observations"] = self._dict["observations"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size

        defaults = {
            "value_p_curgoal": 0.0,
            "value_p_trajgoal": 1.0,
            "value_p_randomgoal": 0.0,
            "value_geom_sample": True,
            "actor_p_curgoal": 0.0,
            "actor_p_trajgoal": 0.5,
            "actor_p_randomgoal": 0.5,
            "actor_geom_sample": True,
            "gc_negative": True,
        }

        self.config["dataset"] = ConfigDict(
            defaults | dict(self.config.get("dataset", {}))
        )

        dataset_config = self.config["dataset"]

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset["terminals"] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            dataset_config["value_p_curgoal"]
            + dataset_config["value_p_trajgoal"]
            + dataset_config["value_p_randomgoal"],
            1.0,
        )
        assert np.isclose(
            dataset_config["actor_p_curgoal"]
            + dataset_config["actor_p_trajgoal"]
            + dataset_config["actor_p_randomgoal"],
            1.0,
        )

        # Set valid_idxs to exclude the final state in each trajectory.
        cur_idx = 0
        valid_idxs = []
        for terminal_idx in self.terminal_locs:
            valid_idxs.append(np.arange(cur_idx, terminal_idx))
            cur_idx = terminal_idx + 1
        self.dataset.valid_idxs = np.concatenate(valid_idxs)

    def sample(self, batch_size, idxs=None, evaluation=False):
        """
        Sample a batch of transitions with goals.

        batch_size: Batch size.
        idxs: Indices of the transitions to sample. If None, random indices are sampled.
        evaluation: Whether to sample for evaluation.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        dataset_config = self.config["dataset"]

        value_goal_idxs = self.sample_goals(
            idxs,
            dataset_config["value_p_curgoal"],
            dataset_config["value_p_trajgoal"],
            dataset_config["value_p_randomgoal"],
            dataset_config["value_geom_sample"],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            dataset_config["actor_p_curgoal"],
            dataset_config["actor_p_trajgoal"],
            dataset_config["actor_p_randomgoal"],
            dataset_config["actor_geom_sample"],
        )

        if "oracle_reps" in self.dataset:
            batch["value_goals"] = self.dataset["oracle_reps"][value_goal_idxs]
            batch["actor_goals"] = self.dataset["oracle_reps"][actor_goal_idxs]
        else:
            batch["value_goals"] = self.get_observations(value_goal_idxs)
            batch["actor_goals"] = self.get_observations(actor_goal_idxs)
        batch["value_goal_observations"] = self.get_observations(value_goal_idxs)
        batch["actor_goal_observations"] = self.get_observations(value_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if dataset_config["gc_negative"] else 0.0)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        if self.config.get("agent_name") in ["trl"]:
            assert (idxs != final_state_idxs).all() and (idxs != value_goal_idxs).all()

            value_middle_goal_idxs = np.random.randint(idxs, value_goal_idxs)

            batch["value_offsets"] = value_goal_idxs - idxs
            batch["value_midpoint_offsets"] = value_middle_goal_idxs - idxs
            batch["value_midpoint_observations"] = self.get_observations(
                value_middle_goal_idxs
            )
            batch["value_midpoint_actions"] = self.dataset["actions"][
                value_middle_goal_idxs
            ]
            batch["next_actions"] = self.dataset["actions"][idxs + 1]

            if "oracle_reps" in self.dataset:
                batch["value_midpoint_goals"] = self.dataset["oracle_reps"][
                    value_middle_goal_idxs
                ]
                batch["value_cur_goals"] = self.dataset["oracle_reps"][idxs]
                batch["value_next_goals"] = self.dataset["oracle_reps"][idxs + 1]
            else:
                batch["value_midpoint_goals"] = self.get_observations(
                    value_middle_goal_idxs
                )
                batch["value_cur_goals"] = self.get_observations(idxs)
                batch["value_next_goals"] = self.get_observations(idxs + 1)

        return batch

    def sample_goals(
        self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, discount=None
    ):
        """Sample goals for the given indices."""
        batch_size = len(idxs)
        if discount is None:
            discount = self.config["discount"]

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(
                p=1 - discount, size=batch_size
            )  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal),
                traj_goal_idxs,
                random_goal_idxs,
            )

            # Goals at the current state.
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs
            )

        return goal_idxs

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        return jax.tree_util.tree_map(
            lambda arr: arr[idxs], self.dataset["observations"]
        )
