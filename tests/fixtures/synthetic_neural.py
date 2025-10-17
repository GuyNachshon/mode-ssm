"""
Synthetic neural data generation for testing MODE-SSM components.
Generates realistic T15-like neural recordings for unit and integration tests.
"""

import numpy as np
import torch
import h5py
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SyntheticTrialData:
    """Container for synthetic neural trial data"""
    neural_features: torch.Tensor  # [T, 512]
    phoneme_labels: torch.Tensor   # [L]
    sequence_length: int
    label_length: int
    session_id: str
    block_num: int
    trial_num: int
    sentence_label: str
    mode_label: int  # 0=silent, 1=vocalized


class SyntheticNeuralDataGenerator:
    """Generate synthetic neural recordings that mimic T15 dataset structure"""

    def __init__(self, random_seed: int = 42):
        """Initialize synthetic data generator

        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_seed)
        self.torch_gen = torch.Generator()
        self.torch_gen.manual_seed(random_seed)

        # T15 dataset constants
        self.num_features = 512
        self.num_electrodes = 256
        self.num_phonemes = 40  # 39 phonemes + blank + silence
        self.sampling_rate = 50.0  # Hz (20ms bins)

        # Synthetic phoneme vocabulary (simplified)
        self.PHONEMES = [
            'BLANK', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D',
            'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH',
            'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S',
            'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
        ]

        # Common test sentences for phoneme generation
        self.TEST_SENTENCES = [
            "hello world",
            "the quick brown fox",
            "neural signals to text",
            "brain computer interface",
            "mode aware decoder",
            "state space models",
            "test time adaptation"
        ]

    def generate_neural_features(
        self,
        duration_ms: int,
        noise_level: float = 0.1,
        missing_channel_prob: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic 512-channel neural features

        Args:
            duration_ms: Trial duration in milliseconds (50-30000)
            noise_level: Gaussian noise standard deviation
            missing_channel_prob: Probability of channel corruption

        Returns:
            neural_features: [T, 512] tensor
            quality_mask: [512] boolean tensor (True = good channel)
        """
        # Convert duration to timesteps (20ms bins)
        n_timesteps = max(3, int(duration_ms / 20))  # Minimum 3 timesteps

        # Generate base neural activity patterns
        # Simulate different electrode arrays with distinct patterns
        features = torch.zeros(n_timesteps, self.num_features)

        for array_idx in range(4):  # 4 electrode arrays
            start_idx = array_idx * 128  # 2 features per electrode × 64 electrodes
            end_idx = start_idx + 128

            # Threshold crossings (first 64 features per array)
            thresh_start = start_idx
            thresh_end = start_idx + 64

            # Generate sparse spike-like activity
            spike_prob = 0.02 + 0.01 * array_idx  # Varying activity across arrays
            spikes = torch.bernoulli(
                torch.full((n_timesteps, 64), spike_prob),
                generator=self.torch_gen
            )
            features[:, thresh_start:thresh_end] = spikes * (2.0 + self.rng.normal(0, 0.5, (n_timesteps, 64)))

            # Spike band power (second 64 features per array)
            power_start = start_idx + 64
            power_end = end_idx

            # Generate power with temporal correlation
            base_power = 1.0 + 0.5 * array_idx
            power_signal = base_power + 0.3 * np.sin(
                2 * np.pi * np.arange(n_timesteps) / (20 + 10 * array_idx)
            )
            power_signal = np.expand_dims(power_signal, 1)
            power_features = power_signal + self.rng.normal(0, 0.2, (n_timesteps, 64))
            features[:, power_start:power_end] = torch.tensor(power_features, dtype=torch.float32)

        # Add global noise
        features += torch.normal(0, noise_level, features.shape, generator=self.torch_gen)

        # Create quality mask (simulate missing/corrupted channels)
        quality_mask = torch.bernoulli(
            torch.full((self.num_features,), 1.0 - missing_channel_prob),
            generator=self.torch_gen
        ).bool()

        # Ensure at least 90% channels are good (as per data validation rules)
        while quality_mask.sum() < int(0.9 * self.num_features):
            bad_indices = (~quality_mask).nonzero(as_tuple=True)[0]
            if len(bad_indices) > 0:
                # Randomly fix some bad channels
                fix_idx = bad_indices[torch.randint(len(bad_indices), (1,), generator=self.torch_gen)]
                quality_mask[fix_idx] = True

        # Zero out bad channels
        features[:, ~quality_mask] = 0.0

        return features, quality_mask

    def text_to_phonemes(self, text: str) -> List[int]:
        """Convert text to phoneme sequence (simplified mapping)

        Args:
            text: Input text string

        Returns:
            List of phoneme indices
        """
        # Simple phoneme mapping (in real implementation would use G2P)
        phoneme_map = {
            'hello': [15, 10, 21, 24],  # HH EH L OW
            'world': [36, 30, 21, 9],   # W ER L D
            'the': [32, 10],            # TH EH
            'quick': [20, 34, 18, 20],  # K UW IY K
            'brown': [7, 28, 24, 23],   # B R OW N
            'fox': [13, 24, 20, 29],    # F OW K S
            'neural': [23, 34, 28, 5],  # N UW R AE
            'test': [31, 10, 29, 31],   # T EH S T
            'brain': [7, 28, 3, 23],    # B R AE N
        }

        words = text.lower().split()
        phonemes = [0]  # Start with blank

        for word in words:
            if word in phoneme_map:
                phonemes.extend(phoneme_map[word])
            else:
                # Default phoneme sequence for unknown words
                phonemes.extend([5, 18, 23, 20])  # AE IY N K
            phonemes.append(0)  # Add blank between words

        return phonemes[:20]  # Limit to reasonable length

    def generate_trial(
        self,
        session_id: str = "2023-01-15",
        block_num: int = 1,
        trial_num: int = 1,
        sentence: Optional[str] = None,
        mode: Optional[int] = None,
        duration_range: Tuple[int, int] = (1000, 5000)  # 1-5 seconds
    ) -> SyntheticTrialData:
        """Generate a complete synthetic trial

        Args:
            session_id: Session identifier
            block_num: Block number
            trial_num: Trial number
            sentence: Text sentence (random if None)
            mode: Speaking mode 0=silent, 1=vocalized (random if None)
            duration_range: (min_ms, max_ms) for trial duration

        Returns:
            SyntheticTrialData object
        """
        # Select or generate sentence
        if sentence is None:
            sentence = self.rng.choice(self.TEST_SENTENCES)

        # Generate speaking mode
        if mode is None:
            mode = self.rng.randint(0, 2)

        # Generate duration
        duration_ms = self.rng.randint(duration_range[0], duration_range[1])

        # Generate neural features
        neural_features, quality_mask = self.generate_neural_features(
            duration_ms=duration_ms,
            noise_level=0.1 + 0.05 * mode,  # Silent has less noise
            missing_channel_prob=0.03 + 0.02 * mode
        )

        # Generate phoneme labels
        phoneme_sequence = self.text_to_phonemes(sentence)
        phoneme_labels = torch.tensor(phoneme_sequence, dtype=torch.long)

        return SyntheticTrialData(
            neural_features=neural_features,
            phoneme_labels=phoneme_labels,
            sequence_length=neural_features.shape[0],
            label_length=len(phoneme_sequence),
            session_id=session_id,
            block_num=block_num,
            trial_num=trial_num,
            sentence_label=sentence,
            mode_label=mode
        )

    def generate_batch(
        self,
        batch_size: int,
        session_prefix: str = "2023-01"
    ) -> List[SyntheticTrialData]:
        """Generate a batch of synthetic trials

        Args:
            batch_size: Number of trials to generate
            session_prefix: Prefix for session IDs

        Returns:
            List of SyntheticTrialData objects
        """
        trials = []
        for i in range(batch_size):
            session_id = f"{session_prefix}-{(i % 30) + 1:02d}"  # 30 days max
            block_num = (i // 10) + 1
            trial_num = (i % 10) + 1

            trial = self.generate_trial(
                session_id=session_id,
                block_num=block_num,
                trial_num=trial_num
            )
            trials.append(trial)

        return trials

    def save_synthetic_hdf5(
        self,
        filename: str,
        trials: List[SyntheticTrialData]
    ) -> None:
        """Save synthetic trials to HDF5 file matching T15 format

        Args:
            filename: Output HDF5 filename
            trials: List of synthetic trials to save
        """
        with h5py.File(filename, 'w') as f:
            for trial in trials:
                # Create group name matching T15 format
                group_name = f"trial_{trial.session_id}_{trial.block_num}_{trial.trial_num}"

                grp = f.create_group(group_name)

                # Neural features
                grp.create_dataset('input_features', data=trial.neural_features.numpy())

                # Phoneme labels
                grp.create_dataset('seq_class_ids', data=trial.phoneme_labels.numpy())

                # Sentence transcription
                grp.create_dataset('transcription',
                                 data=trial.sentence_label.encode('ascii'))

                # Attributes
                grp.attrs['n_time_steps'] = trial.sequence_length
                grp.attrs['seq_len'] = trial.label_length
                grp.attrs['sentence_label'] = trial.sentence_label.encode('ascii')
                grp.attrs['session'] = trial.session_id.encode('ascii')
                grp.attrs['block_num'] = trial.block_num
                grp.attrs['trial_num'] = trial.trial_num


# Convenience functions for tests
def create_synthetic_batch(
    batch_size: int = 8,
    min_seq_len: int = 50,
    max_seq_len: int = 250
) -> Dict[str, torch.Tensor]:
    """Create a synthetic batch for testing

    Args:
        batch_size: Number of samples in batch
        min_seq_len: Minimum sequence length (timesteps)
        max_seq_len: Maximum sequence length (timesteps)

    Returns:
        Dictionary with batch tensors
    """
    generator = SyntheticNeuralDataGenerator()

    # Generate variable length sequences
    neural_features = []
    phoneme_labels = []
    sequence_lengths = []
    label_lengths = []

    max_seq = 0
    max_label = 0

    for i in range(batch_size):
        seq_len = np.random.randint(min_seq_len, max_seq_len)
        trial = generator.generate_trial(duration_range=(seq_len * 20, seq_len * 20))

        neural_features.append(trial.neural_features)
        phoneme_labels.append(trial.phoneme_labels)
        sequence_lengths.append(trial.sequence_length)
        label_lengths.append(trial.label_length)

        max_seq = max(max_seq, trial.sequence_length)
        max_label = max(max_label, trial.label_length)

    # Pad to maximum lengths
    padded_features = torch.zeros(batch_size, max_seq, 512)
    padded_labels = torch.zeros(batch_size, max_label, dtype=torch.long)

    for i in range(batch_size):
        seq_len = neural_features[i].shape[0]
        label_len = phoneme_labels[i].shape[0]

        padded_features[i, :seq_len] = neural_features[i]
        padded_labels[i, :label_len] = phoneme_labels[i]

    return {
        'neural_features': padded_features,
        'phoneme_labels': padded_labels,
        'sequence_lengths': torch.tensor(sequence_lengths),
        'label_lengths': torch.tensor(label_lengths)
    }


def create_test_hdf5(filename: str = "test_data.h5", num_trials: int = 100):
    """Create a test HDF5 file with synthetic data"""
    generator = SyntheticNeuralDataGenerator()
    trials = generator.generate_batch(num_trials)
    generator.save_synthetic_hdf5(filename, trials)
    return filename