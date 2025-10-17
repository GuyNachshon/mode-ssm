"""
T15 Brain-to-Text dataset loading and preprocessing.
Handles HDF5 data format with validation and quality checks.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

from .phoneme_vocab import LOGIT_TO_PHONEME, PHONEME_TO_LOGIT


logger = logging.getLogger(__name__)


@dataclass
class TrialMetadata:
    """Metadata for a single neural trial"""
    session_id: str
    block_num: int
    trial_num: int
    n_time_steps: int
    seq_len: int
    sentence_label: str
    quality_score: float  # Proportion of good channels


class Brain2TextDataset(Dataset):
    """PyTorch Dataset for T15 Brain-to-Text neural recordings"""

    def __init__(
        self,
        hdf5_path: Union[str, Path],
        min_sequence_ms: int = 50,
        max_sequence_ms: int = 30000,
        missing_channel_threshold: float = 0.1,
        cache_data: bool = True,
        transform=None,
        filter_quality: bool = True
    ):
        """Initialize T15 dataset

        Args:
            hdf5_path: Path to HDF5 data file
            min_sequence_ms: Minimum sequence length in milliseconds
            max_sequence_ms: Maximum sequence length in milliseconds
            missing_channel_threshold: Maximum fraction of missing channels allowed
            cache_data: Whether to cache data in memory for faster access
            transform: Optional data transformation pipeline
            filter_quality: Whether to filter out low-quality trials
        """
        self.hdf5_path = Path(hdf5_path)
        self.min_sequence_ms = min_sequence_ms
        self.max_sequence_ms = max_sequence_ms
        self.missing_channel_threshold = missing_channel_threshold
        self.cache_data = cache_data
        self.transform = transform
        self.filter_quality = filter_quality

        # Dataset constants
        self.num_features = 512
        self.num_electrodes = 256
        self.sampling_rate = 50.0  # Hz (20ms bins)
        self.vocab_size = len(LOGIT_TO_PHONEME)

        # Load and validate dataset
        self._load_metadata()
        self._validate_dataset()
        self._cache_data_if_enabled()

        logger.info(f"Loaded {len(self.valid_trials)} valid trials from {self.hdf5_path}")

    def _load_metadata(self):
        """Load trial metadata and filter valid trials"""
        self.all_trials = []
        self.valid_trials = []

        with h5py.File(self.hdf5_path, 'r') as f:
            for trial_key in f.keys():
                if not trial_key.startswith('trial_'):
                    continue

                grp = f[trial_key]

                # Extract metadata - handle both attribute and dataset formats
                try:
                    # Try attrs first (competition format), then datasets (test format)
                    if 'session' in grp.attrs:
                        session_data = grp.attrs['session']
                        if isinstance(session_data, bytes):
                            session_id = session_data.decode('ascii')
                        else:
                            session_id = str(session_data)
                    elif 'session_id' in grp:
                        session_id = str(grp['session_id'][()]) if hasattr(grp['session_id'], '__call__') else str(grp['session_id'][()])
                    else:
                        session_id = 'unknown'

                    if 'block_num' in grp.attrs:
                        block_num = int(grp.attrs['block_num'])
                    elif 'block_id' in grp:
                        block_num = int(grp['block_id'][()])
                    else:
                        block_num = 0

                    if 'trial_num' in grp.attrs:
                        trial_num = int(grp.attrs['trial_num'])
                    elif 'trial_id' in grp:
                        trial_num = int(grp['trial_id'][()])
                    else:
                        trial_num = 0

                    # Get number of timesteps from actual data
                    if 'spikes' in grp:
                        n_time_steps = grp['spikes'].shape[1] if len(grp['spikes'].shape) > 1 else grp['spikes'].shape[0]
                    elif 'input_features' in grp:
                        n_time_steps = grp['input_features'].shape[0]
                    else:
                        n_time_steps = int(grp.attrs.get('n_time_steps', 0))

                    seq_len = int(grp.attrs.get('seq_len', n_time_steps))

                    # Get sentence label - handle both string and bytes formats
                    if 'sentence_label' in grp.attrs:
                        label_data = grp.attrs['sentence_label']
                        if isinstance(label_data, bytes):
                            sentence_label = label_data.decode('ascii')
                        else:
                            sentence_label = str(label_data)
                    else:
                        sentence_label = ''

                    # Get neural features shape for quality check
                    if 'input_features' in grp:
                        neural_features = grp['input_features']
                        expected_dim = 1
                    elif 'spikes' in grp:
                        # Test data format - convert spikes to features
                        neural_features = grp['spikes']
                        expected_dim = 0  # Channels are first dimension
                    else:
                        logger.warning(f"Trial {trial_key}: No neural data found")
                        continue

                    # Check dimensions match
                    feature_dim = neural_features.shape[expected_dim]
                    if feature_dim != self.num_features and feature_dim != self.num_electrodes:
                        logger.warning(f"Trial {trial_key}: Expected {self.num_features} or {self.num_electrodes} features, got {feature_dim}")
                        # Continue anyway for test data
                        if feature_dim < 128:
                            continue

                    # Calculate quality score (assume all channels good if no quality mask)
                    quality_score = self._calculate_quality_score(neural_features[:])

                    metadata = TrialMetadata(
                        session_id=session_id,
                        block_num=block_num,
                        trial_num=trial_num,
                        n_time_steps=n_time_steps,
                        seq_len=seq_len,
                        sentence_label=sentence_label,
                        quality_score=quality_score
                    )

                    self.all_trials.append((trial_key, metadata))

                    # Apply filters
                    if self._is_valid_trial(metadata, n_time_steps):
                        self.valid_trials.append((trial_key, metadata))

                except Exception as e:
                    logger.warning(f"Failed to load trial {trial_key}: {e}")
                    continue

    def _calculate_quality_score(self, neural_features: np.ndarray) -> float:
        """Calculate quality score based on missing/corrupted channels

        Args:
            neural_features: [T, 512] neural feature array

        Returns:
            Quality score (0-1, higher is better)
        """
        # Check for zero/constant channels (indicate missing data)
        channel_means = np.mean(neural_features, axis=0)
        channel_stds = np.std(neural_features, axis=0)

        # Channels with very low std are likely corrupted/missing
        good_channels = (channel_stds > 1e-6) & (np.abs(channel_means) < 1e3)
        return float(np.mean(good_channels))

    def _is_valid_trial(self, metadata: TrialMetadata, n_time_steps: int) -> bool:
        """Check if trial passes quality filters

        Args:
            metadata: Trial metadata
            n_time_steps: Number of timesteps

        Returns:
            True if trial is valid
        """
        # Duration limits (convert timesteps to ms)
        duration_ms = n_time_steps * (1000 / self.sampling_rate)
        if duration_ms < self.min_sequence_ms or duration_ms > self.max_sequence_ms:
            return False

        # Quality threshold
        if self.filter_quality and metadata.quality_score < (1.0 - self.missing_channel_threshold):
            return False

        # For training data, require valid sequence length
        # For test data without labels, seq_len can be 0
        # Skip this check for now to allow test data
        # if metadata.seq_len <= 0:
        #     return False

        return True

    def _validate_dataset(self):
        """Validate dataset format and content"""
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.hdf5_path}")

        if len(self.valid_trials) == 0:
            raise ValueError(f"No valid trials found in {self.hdf5_path}")

        # Check data format with first valid trial
        with h5py.File(self.hdf5_path, 'r') as f:
            trial_key, _ = self.valid_trials[0]
            grp = f[trial_key]

            # Validate required fields - accept either format
            if 'input_features' in grp:
                neural_features = grp['input_features']
                if len(neural_features.shape) != 2:
                    raise ValueError(f"Invalid neural features shape: {neural_features.shape}")
            elif 'spikes' in grp:
                neural_features = grp['spikes']
                if len(neural_features.shape) != 2:
                    raise ValueError(f"Invalid spikes shape: {neural_features.shape}")
            else:
                raise ValueError("Missing neural data (input_features or spikes)")

        logger.info(f"Dataset validation passed: {len(self.valid_trials)}/{len(self.all_trials)} valid trials")

    def _cache_data_if_enabled(self):
        """Cache data in memory if enabled"""
        if not self.cache_data:
            self.cached_data = None
            return

        logger.info("Caching dataset in memory...")
        self.cached_data = {}

        with h5py.File(self.hdf5_path, 'r') as f:
            for i, (trial_key, metadata) in enumerate(self.valid_trials):
                grp = f[trial_key]

                # Load neural features - handle both formats
                if 'input_features' in grp:
                    neural_features = torch.tensor(grp['input_features'][:], dtype=torch.float32)
                elif 'spikes' in grp:
                    # Convert spikes (channels, time) to features (time, channels)
                    spikes = grp['spikes'][:]
                    neural_features = torch.tensor(spikes.T, dtype=torch.float32)

                # Load phoneme labels if available
                phoneme_labels = None
                if 'seq_class_ids' in grp:
                    phoneme_labels = torch.tensor(grp['seq_class_ids'][:], dtype=torch.long)

                # Load transcription if available
                transcription = None
                if 'transcription' in grp:
                    trans_data = grp['transcription'][:]
                    if isinstance(trans_data, np.ndarray) and trans_data.dtype.kind in ['u', 'i']:
                        # ASCII codes array - convert to string
                        transcription = ''.join(chr(c) for c in trans_data if c != 0)
                    elif isinstance(trans_data, bytes):
                        transcription = trans_data.decode('ascii')
                    else:
                        transcription = str(trans_data)

                self.cached_data[i] = {
                    'neural_features': neural_features,
                    'phoneme_labels': phoneme_labels,
                    'transcription': transcription,
                    'metadata': metadata
                }

        logger.info(f"Cached {len(self.cached_data)} trials")

    def __len__(self) -> int:
        return len(self.valid_trials)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single trial

        Args:
            idx: Trial index

        Returns:
            Dictionary with trial data
        """
        if self.cache_data and self.cached_data is not None:
            return self._get_cached_item(idx)
        else:
            return self._get_item_from_disk(idx)

    def _get_cached_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from cached data"""
        data = self.cached_data[idx]

        item = {
            'neural_features': data['neural_features'],
            'sequence_length': torch.tensor(data['neural_features'].shape[0], dtype=torch.long),
            'session_id': data['metadata'].session_id,
            'block_num': torch.tensor(data['metadata'].block_num, dtype=torch.long),
            'trial_num': torch.tensor(data['metadata'].trial_num, dtype=torch.long),
        }

        if data['phoneme_labels'] is not None:
            item['phoneme_labels'] = data['phoneme_labels']
            item['label_length'] = torch.tensor(len(data['phoneme_labels']), dtype=torch.long)

        if data['transcription'] is not None:
            item['transcription'] = data['transcription']

        # Apply transforms
        if self.transform is not None:
            item = self.transform(item)

        return item

    def _get_item_from_disk(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item directly from HDF5 file"""
        trial_key, metadata = self.valid_trials[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            grp = f[trial_key]

            # Load neural features - handle both formats
            if 'input_features' in grp:
                neural_features = torch.tensor(grp['input_features'][:], dtype=torch.float32)
            elif 'spikes' in grp:
                # Convert spikes (channels, time) to features (time, channels)
                spikes = grp['spikes'][:]
                neural_features = torch.tensor(spikes.T, dtype=torch.float32)

            item = {
                'neural_features': neural_features,
                'sequence_length': torch.tensor(neural_features.shape[0], dtype=torch.long),
                'session_id': metadata.session_id,
                'block_num': torch.tensor(metadata.block_num, dtype=torch.long),
                'trial_num': torch.tensor(metadata.trial_num, dtype=torch.long),
            }

            # Load phoneme labels if available
            if 'seq_class_ids' in grp:
                phoneme_labels = torch.tensor(grp['seq_class_ids'][:], dtype=torch.long)
                item['phoneme_labels'] = phoneme_labels
                item['label_length'] = torch.tensor(len(phoneme_labels), dtype=torch.long)

            # Load transcription if available
            if 'transcription' in grp:
                trans_data = grp['transcription'][:]
                if isinstance(trans_data, np.ndarray) and trans_data.dtype.kind in ['u', 'i']:
                    # ASCII codes array - convert to string
                    item['transcription'] = ''.join(chr(c) for c in trans_data if c != 0)
                elif isinstance(trans_data, bytes):
                    item['transcription'] = trans_data.decode('ascii')
                else:
                    item['transcription'] = str(trans_data)

        # Apply transforms
        if self.transform is not None:
            item = self.transform(item)

        return item

    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_trials': len(self.all_trials),
            'valid_trials': len(self.valid_trials),
            'quality_filtered': len(self.all_trials) - len(self.valid_trials),
        }

        if len(self.valid_trials) > 0:
            sequence_lengths = [metadata.n_time_steps for _, metadata in self.valid_trials]
            label_lengths = [metadata.seq_len for _, metadata in self.valid_trials]
            quality_scores = [metadata.quality_score for _, metadata in self.valid_trials]

            stats.update({
                'mean_sequence_length': float(np.mean(sequence_lengths)),
                'std_sequence_length': float(np.std(sequence_lengths)),
                'mean_label_length': float(np.mean(label_lengths)),
                'std_label_length': float(np.std(label_lengths)),
                'mean_quality_score': float(np.mean(quality_scores)),
                'min_quality_score': float(np.min(quality_scores)),
            })

        return stats

    def get_sessions(self) -> List[str]:
        """Get list of unique session IDs"""
        return list(set(metadata.session_id for _, metadata in self.valid_trials))

    def filter_by_session(self, session_ids: List[str]) -> 'Brain2TextDataset':
        """Create a new dataset filtered by session IDs"""
        # This would create a subset - implementation depends on specific use case
        raise NotImplementedError("Session filtering not implemented")


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader with variable-length sequences

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors with padding
    """
    # Find maximum lengths
    max_seq_len = max(item['sequence_length'].item() for item in batch)
    max_label_len = 0
    has_labels = 'phoneme_labels' in batch[0] and batch[0]['phoneme_labels'] is not None

    if has_labels:
        max_label_len = max(item['label_length'].item() for item in batch if 'label_length' in item)

    batch_size = len(batch)

    # Get number of channels from first item
    num_channels = batch[0]['neural_features'].shape[1]

    # Initialize padded tensors
    neural_features = torch.zeros(batch_size, max_seq_len, num_channels, dtype=torch.float32)
    sequence_lengths = torch.stack([item['sequence_length'] for item in batch])

    # Optional tensors
    phoneme_labels = None
    label_lengths = None
    if has_labels and max_label_len > 0:
        phoneme_labels = torch.zeros(batch_size, max_label_len, dtype=torch.long)
        label_lengths = torch.stack([item.get('label_length', torch.tensor(0)) for item in batch])

    # Fill padded tensors
    for i, item in enumerate(batch):
        seq_len = item['sequence_length'].item()
        neural_features[i, :seq_len] = item['neural_features']

        if has_labels and phoneme_labels is not None:
            label_len = item.get('label_length', torch.tensor(0)).item()
            if label_len > 0:
                phoneme_labels[i, :label_len] = item['phoneme_labels']

    # Build batch dictionary
    batched = {
        'neural_features': neural_features,
        'sequence_lengths': sequence_lengths,
        'session_ids': [item['session_id'] for item in batch],
        'block_nums': torch.stack([item['block_num'] for item in batch]),
        'trial_nums': torch.stack([item['trial_num'] for item in batch]),
    }

    if phoneme_labels is not None:
        batched['phoneme_labels'] = phoneme_labels
        batched['label_lengths'] = label_lengths

    # Add transcriptions if available
    if 'transcription' in batch[0]:
        batched['transcriptions'] = [item.get('transcription', '') for item in batch]

    return batched


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders

    Args:
        train_path: Path to training HDF5 file
        val_path: Path to validation HDF5 file
        test_path: Optional path to test HDF5 file
        batch_size: Training batch size
        val_batch_size: Validation batch size (defaults to batch_size)
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for GPU transfer
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        Dictionary with train/val/test dataloaders
    """
    if val_batch_size is None:
        val_batch_size = batch_size

    # Create datasets
    train_dataset = Brain2TextDataset(train_path, **dataset_kwargs)
    val_dataset = Brain2TextDataset(val_path, **dataset_kwargs)

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            drop_last=False
        )
    }

    if test_path is not None:
        test_dataset = Brain2TextDataset(test_path, **dataset_kwargs)
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            drop_last=False
        )

    return dataloaders