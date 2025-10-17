"""
Distributed data sampler for multi-GPU training.
Handles data distribution across processes with proper shuffling and epoch management.
"""

import torch
import torch.distributed as dist
from torch.utils.data import Sampler, Dataset
import math
import random
from typing import Iterator, Optional, TypeVar, List, Dict
import numpy as np


T_co = TypeVar('T_co', covariant=True)


class DistributedSamplerForBrainData(Sampler[T_co]):
    """
    Distributed sampler specifically designed for brain-to-text data.

    Features:
    - Ensures each GPU gets unique subset of data
    - Supports variable-length sequences
    - Session-aware sampling
    - Proper shuffling with seed management
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        session_balanced: bool = False
    ):
        """
        Initialize distributed sampler.

        Args:
            dataset: Dataset to sample from
            num_replicas: Number of processes (GPUs)
            rank: Rank of current process
            shuffle: Whether to shuffle indices
            seed: Random seed for shuffling
            drop_last: Whether to drop last incomplete batch
            session_balanced: Whether to balance sessions across GPUs
        """
        if num_replicas is None:
            if not dist.is_available():
                num_replicas = 1
            elif not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                rank = 0
            elif not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, should be in [0, {num_replicas})"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.session_balanced = session_balanced

        # Calculate samples per replica
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Drop samples to make it evenly divisible
            self.num_samples = math.floor(len(self.dataset) / self.num_replicas)
        else:
            # Add extra samples to make it evenly divisible
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

        # Session information if available
        self.session_indices = self._get_session_indices() if session_balanced else None

    def _get_session_indices(self) -> Optional[Dict[str, List[int]]]:
        """Get indices grouped by session if available"""
        if not hasattr(self.dataset, 'get_sessions'):
            return None

        sessions = self.dataset.get_sessions()
        session_indices = {session: [] for session in sessions}

        for idx in range(len(self.dataset)):
            if hasattr(self.dataset, 'valid_trials'):
                _, metadata = self.dataset.valid_trials[idx]
                session = metadata.session_id
                if session in session_indices:
                    session_indices[session].append(idx)
            else:
                # Fallback: distribute evenly
                session = sessions[idx % len(sessions)]
                session_indices[session].append(idx)

        return session_indices

    def __iter__(self) -> Iterator[T_co]:
        """Iterate over sampled indices"""
        if self.shuffle:
            # Deterministic shuffling based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            if self.session_balanced and self.session_indices:
                indices = self._balanced_shuffle(g)
            else:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Handle dataset size not divisible by num_replicas
        if not self.drop_last:
            # Add extra indices to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail to make it evenly divisible
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # Subsample for current rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def _balanced_shuffle(self, generator: torch.Generator) -> List[int]:
        """
        Shuffle indices while trying to balance sessions across GPUs.
        """
        all_indices = []

        # Shuffle within each session
        for session, session_idx in self.session_indices.items():
            session_tensor = torch.tensor(session_idx)
            shuffled = session_tensor[torch.randperm(len(session_tensor), generator=generator)]
            all_indices.extend(shuffled.tolist())

        # Shuffle sessions themselves
        final_indices = torch.tensor(all_indices)
        final_indices = final_indices[torch.randperm(len(final_indices), generator=generator)]

        return final_indices.tolist()

    def __len__(self) -> int:
        """Return number of samples for this rank"""
        return self.num_samples

    def set_epoch(self, epoch: int):
        """
        Set epoch for shuffling.
        Should be called at the beginning of each epoch.

        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch


class SequenceLengthSampler(Sampler[T_co]):
    """
    Sampler that groups sequences by length to minimize padding.
    Useful for efficient training with variable-length sequences.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        bucket_boundaries: Optional[List[int]] = None
    ):
        """
        Initialize sequence length sampler.

        Args:
            dataset: Dataset with sequence_length attribute
            batch_size: Batch size for grouping
            shuffle: Whether to shuffle within buckets
            seed: Random seed
            drop_last: Whether to drop last incomplete batch
            bucket_boundaries: Boundaries for length buckets
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Get sequence lengths
        self.lengths = self._get_sequence_lengths()

        # Create buckets
        if bucket_boundaries is None:
            # Default boundaries: quantiles
            bucket_boundaries = np.quantile(self.lengths, [0.25, 0.5, 0.75]).tolist()

        self.bucket_boundaries = [0] + bucket_boundaries + [float('inf')]
        self.buckets = self._create_buckets()

    def _get_sequence_lengths(self) -> List[int]:
        """Get sequence lengths from dataset"""
        lengths = []
        for idx in range(len(self.dataset)):
            if hasattr(self.dataset, '__getitem__'):
                item = self.dataset[idx]
                if 'sequence_length' in item:
                    lengths.append(item['sequence_length'].item())
                else:
                    lengths.append(100)  # Default length
            else:
                lengths.append(100)  # Default length

        return lengths

    def _create_buckets(self) -> Dict[int, List[int]]:
        """Create buckets of indices based on sequence length"""
        buckets = {i: [] for i in range(len(self.bucket_boundaries) - 1)}

        for idx, length in enumerate(self.lengths):
            # Find appropriate bucket
            for bucket_idx in range(len(self.bucket_boundaries) - 1):
                if self.bucket_boundaries[bucket_idx] <= length < self.bucket_boundaries[bucket_idx + 1]:
                    buckets[bucket_idx].append(idx)
                    break

        return buckets

    def __iter__(self) -> Iterator[T_co]:
        """Iterate over indices grouped by length"""
        # Set random seed based on epoch
        if self.shuffle:
            random.seed(self.seed + self.epoch)

        all_batches = []

        # Create batches from each bucket
        for bucket_idx, indices in self.buckets.items():
            if len(indices) == 0:
                continue

            # Shuffle indices within bucket
            if self.shuffle:
                random.shuffle(indices)

            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # Shuffle batches
        if self.shuffle:
            random.shuffle(all_batches)

        # Flatten batches
        for batch in all_batches:
            for idx in batch:
                yield idx

    def __len__(self) -> int:
        """Return total number of samples"""
        total = 0
        for indices in self.buckets.values():
            if self.drop_last:
                total += (len(indices) // self.batch_size) * self.batch_size
            else:
                total += len(indices)
        return total

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling"""
        self.epoch = epoch


class DistributedSequenceLengthSampler(Sampler[T_co]):
    """
    Combines distributed sampling with sequence length bucketing.
    Ensures efficient padding while distributing data across GPUs.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ):
        """
        Initialize distributed sequence length sampler.

        Args:
            dataset: Dataset to sample from
            batch_size: Batch size for bucketing
            num_replicas: Number of processes
            rank: Current process rank
            shuffle: Whether to shuffle
            seed: Random seed
            drop_last: Whether to drop last batch
        """
        # Initialize distributed parameters
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Create underlying sequence length sampler
        self.length_sampler = SequenceLengthSampler(
            dataset=dataset,
            batch_size=batch_size * num_replicas,  # Scale batch size
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )

        # Calculate samples per replica
        total_samples = len(self.length_sampler)
        self.num_samples = math.ceil(total_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[T_co]:
        """Iterate over distributed length-bucketed indices"""
        # Get all indices from length sampler
        self.length_sampler.set_epoch(self.epoch)
        all_indices = list(self.length_sampler)

        # Pad or truncate to make divisible
        if not self.drop_last and len(all_indices) < self.total_size:
            # Pad with repeated indices
            padding_size = self.total_size - len(all_indices)
            all_indices += all_indices[:padding_size]
        elif self.drop_last:
            all_indices = all_indices[:self.total_size]

        # Select indices for current rank
        rank_indices = all_indices[self.rank::self.num_replicas]

        return iter(rank_indices)

    def __len__(self) -> int:
        """Return number of samples for this rank"""
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling"""
        self.epoch = epoch
        self.length_sampler.set_epoch(epoch)


def create_distributed_sampler(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
    use_length_bucketing: bool = True,
    session_balanced: bool = False
) -> Sampler:
    """
    Create appropriate distributed sampler based on configuration.

    Args:
        dataset: Dataset to sample from
        batch_size: Batch size
        shuffle: Whether to shuffle
        seed: Random seed
        drop_last: Whether to drop last batch
        use_length_bucketing: Whether to bucket by sequence length
        session_balanced: Whether to balance sessions across GPUs

    Returns:
        Configured sampler
    """
    # Check if distributed training is available
    distributed = dist.is_available() and dist.is_initialized()

    if distributed:
        if use_length_bucketing:
            return DistributedSequenceLengthSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last
            )
        else:
            return DistributedSamplerForBrainData(
                dataset=dataset,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
                session_balanced=session_balanced
            )
    else:
        # Single GPU training
        if use_length_bucketing:
            return SequenceLengthSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last
            )
        else:
            # Standard random sampling
            return None  # Use default DataLoader sampler