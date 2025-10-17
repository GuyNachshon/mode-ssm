"""
Multi-session dataset loader for T15 Brain-to-Text data.
Combines multiple session HDF5 files into a single dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from torch.utils.data import ConcatDataset, DataLoader

from .brain2text import Brain2TextDataset, collate_batch

logger = logging.getLogger(__name__)


class MultiSessionDataset(ConcatDataset):
    """
    Dataset that combines multiple session HDF5 files.

    Wrapper around PyTorch's ConcatDataset that creates individual
    Brain2TextDataset instances for each session file.
    """

    def __init__(
        self,
        session_paths: List[Union[str, Path]],
        **dataset_kwargs
    ):
        """
        Initialize multi-session dataset.

        Args:
            session_paths: List of paths to session HDF5 files
            **dataset_kwargs: Additional arguments passed to Brain2TextDataset
        """
        self.session_paths = [Path(p) for p in session_paths]

        # Create individual datasets for each session
        self.session_datasets = []
        for session_path in self.session_paths:
            if not session_path.exists():
                logger.warning(f"Session file not found: {session_path}")
                continue

            try:
                dataset = Brain2TextDataset(session_path, **dataset_kwargs)
                self.session_datasets.append(dataset)
                logger.info(f"Loaded session {session_path.parent.name}: {len(dataset)} trials")
            except Exception as e:
                logger.warning(f"Failed to load session {session_path}: {e}")
                continue

        if len(self.session_datasets) == 0:
            raise ValueError("No valid session datasets loaded")

        # Initialize ConcatDataset with all session datasets
        super().__init__(self.session_datasets)

        # Compute statistics
        total_trials = sum(len(ds) for ds in self.session_datasets)
        logger.info(
            f"MultiSessionDataset initialized: {len(self.session_datasets)} sessions, "
            f"{total_trials} total trials"
        )

    def get_session_info(self) -> Dict[str, int]:
        """Get trial counts per session"""
        return {
            str(path.parent.name): len(ds)
            for path, ds in zip(self.session_paths, self.session_datasets)
        }


def collect_session_files(
    data_root: Union[str, Path],
    split: str = "train",
    limit: Optional[int] = None
) -> List[Path]:
    """
    Collect all session HDF5 files for a given split.

    Args:
        data_root: Root directory containing session folders (e.g., hdf5_data_final/)
        split: Data split ("train", "val", or "test")
        limit: Optional limit on number of sessions to load

    Returns:
        List of paths to HDF5 files
    """
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Find all session directories (format: t15.YYYY.MM.DD)
    session_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith('t15.')
    ])

    if limit is not None:
        session_dirs = session_dirs[:limit]

    # Collect data files for the specified split
    session_files = []
    for session_dir in session_dirs:
        data_file = session_dir / f"data_{split}.hdf5"
        if data_file.exists():
            session_files.append(data_file)
        else:
            logger.warning(f"Missing {split} file for session {session_dir.name}")

    logger.info(f"Found {len(session_files)} {split} files across {len(session_dirs)} sessions")

    return session_files


def create_multi_session_dataloaders(
    data_root: Union[str, Path],
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_sessions: Optional[int] = None,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create dataloaders from all sessions in a directory.

    Args:
        data_root: Root directory containing session folders
        batch_size: Training batch size
        val_batch_size: Validation batch size (defaults to batch_size)
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for GPU transfer
        max_sessions: Optional limit on number of sessions
        **dataset_kwargs: Additional arguments for Brain2TextDataset

    Returns:
        Dictionary with train/val/test dataloaders
    """
    if val_batch_size is None:
        val_batch_size = batch_size

    # Collect session files
    train_files = collect_session_files(data_root, "train", max_sessions)
    val_files = collect_session_files(data_root, "val", max_sessions)
    test_files = collect_session_files(data_root, "test", max_sessions)

    # Create multi-session datasets
    train_dataset = MultiSessionDataset(train_files, **dataset_kwargs)
    val_dataset = MultiSessionDataset(val_files, **dataset_kwargs)

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=False
        )
    }

    # Add test dataloader if test files exist
    if len(test_files) > 0:
        test_dataset = MultiSessionDataset(test_files, **dataset_kwargs)
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=False
        )

    return dataloaders
