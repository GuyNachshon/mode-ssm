"""
Create synthetic test data for MODE-SSM training.
This generates mock neural signal data for testing the pipeline.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def create_synthetic_data():
    """Create synthetic neural signal data for testing."""

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Parameters
    num_sessions = 2
    num_trials_per_session = 10
    num_channels = 256
    sample_rate = 2000  # Hz
    duration_ms = 2000  # 2 seconds
    time_bins = int(duration_ms * sample_rate / 1000)

    # Sample phonemes and words
    phonemes = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
                'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K']
    words = ['HELLO', 'WORLD', 'TEST', 'DATA', 'NEURAL', 'SIGNAL', 'MODE', 'SSM']

    # Create training data
    train_file = data_dir / "train.h5"
    with h5py.File(train_file, 'w') as f:
        trial_idx = 0

        for session in range(num_sessions):
            for trial in range(num_trials_per_session):
                # Generate synthetic neural signals
                # Add some structure: channels have different base activities
                base_activity = np.random.exponential(0.5, (num_channels, 1))
                noise = np.random.poisson(base_activity * np.ones((num_channels, time_bins)))

                # Add some correlated activity patterns (simulating real neural activity)
                for pattern in range(5):
                    start_time = np.random.randint(0, time_bins - 200)
                    end_time = start_time + np.random.randint(100, 200)
                    pattern_channels = np.random.choice(num_channels, 20, replace=False)
                    for ch in pattern_channels:
                        noise[ch, start_time:end_time] += np.random.poisson(2, end_time - start_time)

                # Store trial data
                trial_group = f"trial_{trial_idx:04d}"
                f.create_dataset(f"{trial_group}/spikes", data=noise.astype(np.int16))

                # Create synthetic phoneme sequence
                num_phonemes = np.random.randint(3, 8)
                phoneme_seq = np.random.choice(phonemes, num_phonemes)
                phoneme_str = " ".join(phoneme_seq)

                # Create phoneme IDs (1-based, reserve 0 for blank)
                phoneme_ids = np.random.randint(1, 40, num_phonemes)

                # Create synthetic text (approximate)
                word = np.random.choice(words)

                # Store labels
                f.create_dataset(f"{trial_group}/phonemes",
                               data=phoneme_str.encode('utf-8'))
                f.create_dataset(f"{trial_group}/text",
                               data=word.encode('utf-8'))
                f.create_dataset(f"{trial_group}/seq_class_ids",
                               data=phoneme_ids.astype(np.int32))

                # Store metadata
                f.create_dataset(f"{trial_group}/session_id", data=session)
                f.create_dataset(f"{trial_group}/trial_id", data=trial)
                f.create_dataset(f"{trial_group}/block_id", data=0)
                f.create_dataset(f"{trial_group}/duration_ms", data=duration_ms)
                f.create_dataset(f"{trial_group}/sample_rate", data=sample_rate)

                # Speaking mode (0=silent, 1=overt)
                mode = np.random.randint(0, 2)
                f.create_dataset(f"{trial_group}/mode", data=mode)

                trial_idx += 1

        # Store global metadata
        f.attrs['num_trials'] = trial_idx
        f.attrs['num_channels'] = num_channels
        f.attrs['sample_rate'] = sample_rate
        f.attrs['phoneme_vocab'] = [p.encode('utf-8') for p in phonemes]

    # Create validation data (smaller)
    val_file = data_dir / "val.h5"
    with h5py.File(val_file, 'w') as f:
        trial_idx = 0

        for session in range(1):  # Single session for validation
            for trial in range(5):  # Fewer trials
                # Generate synthetic neural signals
                base_activity = np.random.exponential(0.5, (num_channels, 1))
                noise = np.random.poisson(base_activity * np.ones((num_channels, time_bins)))

                # Store trial data
                trial_group = f"trial_{trial_idx:04d}"
                f.create_dataset(f"{trial_group}/spikes", data=noise.astype(np.int16))

                # Create synthetic labels
                num_val_phonemes = np.random.randint(3, 6)
                phoneme_seq = np.random.choice(phonemes, num_val_phonemes)
                phoneme_str = " ".join(phoneme_seq)
                phoneme_ids = np.random.randint(1, 40, num_val_phonemes)
                word = np.random.choice(words)

                f.create_dataset(f"{trial_group}/phonemes",
                               data=phoneme_str.encode('utf-8'))
                f.create_dataset(f"{trial_group}/text",
                               data=word.encode('utf-8'))
                f.create_dataset(f"{trial_group}/seq_class_ids",
                               data=phoneme_ids.astype(np.int32))

                # Store metadata
                f.create_dataset(f"{trial_group}/session_id", data=session)
                f.create_dataset(f"{trial_group}/trial_id", data=trial)
                f.create_dataset(f"{trial_group}/block_id", data=0)
                f.create_dataset(f"{trial_group}/duration_ms", data=duration_ms)
                f.create_dataset(f"{trial_group}/sample_rate", data=sample_rate)

                mode = np.random.randint(0, 2)
                f.create_dataset(f"{trial_group}/mode", data=mode)

                trial_idx += 1

        f.attrs['num_trials'] = trial_idx
        f.attrs['num_channels'] = num_channels
        f.attrs['sample_rate'] = sample_rate
        f.attrs['phoneme_vocab'] = [p.encode('utf-8') for p in phonemes]

    print(f"Created synthetic training data: {train_file}")
    print(f"Created synthetic validation data: {val_file}")
    print(f"Training trials: {num_sessions * num_trials_per_session}")
    print(f"Validation trials: 5")
    print(f"Channels: {num_channels}, Duration: {duration_ms}ms")

if __name__ == "__main__":
    create_synthetic_data()