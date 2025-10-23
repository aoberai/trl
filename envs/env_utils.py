import numpy as np
import ogbench

from utils.datasets import Dataset
from utils.evaluation import EpisodeMonitor


def make_env_and_datasets(dataset_name, dataset_path, dataset_only=False, cur_env=None):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the environment (dataset).
        dataset_path: Path to the dataset file.
        dataset_only: Whether to return only the datasets.
        cur_env: Current environment (only used when `dataset_only` is True).

    Returns:
        A tuple of the environment (if `dataset_only` is False), training dataset, and validation dataset.
    """

    if dataset_only:
        train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name,
            dataset_path=dataset_path,
            compact_dataset=True,
            dataset_only=dataset_only,
            cur_env=cur_env,
        )
    else:
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name,
            dataset_path=dataset_path,
            compact_dataset=True,
            dataset_only=dataset_only,
            cur_env=cur_env,
        )
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    # Clip dataset actions.
    eps = 1e-5
    train_dataset = train_dataset.copy(
        add_or_replace=dict(
            actions=np.clip(train_dataset["actions"], -1 + eps, 1 - eps)
        )
    )
    val_dataset = val_dataset.copy(
        add_or_replace=dict(actions=np.clip(val_dataset["actions"], -1 + eps, 1 - eps))
    )

    if dataset_only:
        return train_dataset, val_dataset
    else:
        env.reset()
        return env, train_dataset, val_dataset
