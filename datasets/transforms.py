"""
Data augmentation transforms for neural signal processing.
Implements temporal masking, time warping, and other augmentations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import random


class TemporalMasking(nn.Module):
    """
    Apply temporal masking to neural signals for augmentation.
    Randomly masks consecutive time steps to improve robustness.
    """

    def __init__(
        self,
        mask_prob: float = 0.1,
        max_mask_length: int = 10,
        mask_value: float = 0.0,
        num_masks: int = 2
    ):
        """
        Initialize temporal masking.

        Args:
            mask_prob: Probability of applying masking
            max_mask_length: Maximum length of each mask
            mask_value: Value to use for masked positions
            num_masks: Number of masks to apply
        """
        super().__init__()
        self.mask_prob = mask_prob
        self.max_mask_length = max_mask_length
        self.mask_value = mask_value
        self.num_masks = num_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal masking.

        Args:
            x: Input tensor [batch_size, seq_len, num_channels]

        Returns:
            Masked tensor [batch_size, seq_len, num_channels]
        """
        if not self.training or random.random() > self.mask_prob:
            return x

        batch_size, seq_len, num_channels = x.shape
        x_masked = x.clone()

        for b in range(batch_size):
            for _ in range(self.num_masks):
                # Random mask length
                mask_length = random.randint(1, min(self.max_mask_length, seq_len))

                # Random start position
                max_start = seq_len - mask_length
                if max_start > 0:
                    start = random.randint(0, max_start)
                else:
                    start = 0

                # Apply mask
                x_masked[b, start:start + mask_length, :] = self.mask_value

        return x_masked


class TimeWarping(nn.Module):
    """
    Apply time warping augmentation to neural signals.
    Stretches or compresses segments of the signal in time.
    """

    def __init__(
        self,
        warp_prob: float = 0.1,
        max_warp_factor: float = 1.2,
        num_control_points: int = 4
    ):
        """
        Initialize time warping.

        Args:
            warp_prob: Probability of applying warping
            max_warp_factor: Maximum warping factor (>1 stretches, <1 compresses)
            num_control_points: Number of control points for warping
        """
        super().__init__()
        self.warp_prob = warp_prob
        self.max_warp_factor = max_warp_factor
        self.num_control_points = num_control_points

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping.

        Args:
            x: Input tensor [batch_size, seq_len, num_channels]

        Returns:
            Warped tensor [batch_size, seq_len, num_channels]
        """
        if not self.training or random.random() > self.warp_prob:
            return x

        batch_size, seq_len, num_channels = x.shape
        device = x.device

        # Create warping grid
        grid = self._create_warping_grid(seq_len, device)

        # Apply warping to each sample in batch
        x_warped = []
        for b in range(batch_size):
            # Reshape for grid_sample: [1, C, 1, T]
            x_b = x[b].permute(1, 0).unsqueeze(0).unsqueeze(2)  # [1, C, 1, T]

            # Apply warping
            warped = F.grid_sample(
                x_b,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )

            # Reshape back: [T, C]
            warped = warped.squeeze(0).squeeze(1).permute(1, 0)
            x_warped.append(warped)

        return torch.stack(x_warped, dim=0)

    def _create_warping_grid(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create warping grid for time warping"""
        # Generate control points
        control_points = torch.linspace(0, seq_len - 1, self.num_control_points + 2)

        # Add random perturbations to interior control points
        for i in range(1, len(control_points) - 1):
            warp_factor = 1.0 + (random.random() * 2 - 1) * (self.max_warp_factor - 1)
            control_points[i] = control_points[i] * warp_factor

        # Ensure monotonicity (no time reversal)
        for i in range(1, len(control_points)):
            control_points[i] = max(control_points[i], control_points[i - 1] + 1)

        # Normalize control points
        control_points = control_points / (seq_len - 1) * 2 - 1  # Range [-1, 1]

        # Interpolate to full sequence length
        orig_points = torch.linspace(-1, 1, len(control_points))
        target_points = torch.linspace(-1, 1, seq_len)

        # Linear interpolation
        warped_points = torch.zeros(seq_len)
        for i, t in enumerate(target_points):
            # Find surrounding control points
            idx = torch.searchsorted(orig_points, t)
            if idx == 0:
                warped_points[i] = control_points[0]
            elif idx >= len(orig_points):
                warped_points[i] = control_points[-1]
            else:
                # Linear interpolation
                alpha = (t - orig_points[idx - 1]) / (orig_points[idx] - orig_points[idx - 1])
                warped_points[i] = (1 - alpha) * control_points[idx - 1] + alpha * control_points[idx]

        # Create grid for grid_sample
        grid = torch.zeros(1, 1, seq_len, 2, device=device)
        grid[0, 0, :, 0] = warped_points.to(device)  # x coordinates (time)
        grid[0, 0, :, 1] = 0  # y coordinates (not used for 1D)

        return grid


class ChannelDropout(nn.Module):
    """
    Randomly drop entire channels (electrodes) during training.
    Simulates electrode failures and improves robustness.
    """

    def __init__(
        self,
        drop_prob: float = 0.1,
        max_channels_to_drop: int = 50
    ):
        """
        Initialize channel dropout.

        Args:
            drop_prob: Probability of applying channel dropout
            max_channels_to_drop: Maximum number of channels to drop
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.max_channels_to_drop = max_channels_to_drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel dropout.

        Args:
            x: Input tensor [batch_size, seq_len, num_channels]

        Returns:
            Tensor with dropped channels [batch_size, seq_len, num_channels]
        """
        if not self.training or random.random() > self.drop_prob:
            return x

        batch_size, seq_len, num_channels = x.shape
        x_dropped = x.clone()

        for b in range(batch_size):
            # Random number of channels to drop
            num_drop = random.randint(1, min(self.max_channels_to_drop, num_channels // 4))

            # Random channels to drop
            channels_to_drop = random.sample(range(num_channels), num_drop)

            # Drop channels
            x_dropped[b, :, channels_to_drop] = 0.0

        return x_dropped


class GaussianNoise(nn.Module):
    """
    Add Gaussian noise to neural signals.
    """

    def __init__(
        self,
        noise_prob: float = 0.1,
        noise_std: float = 0.1
    ):
        """
        Initialize Gaussian noise.

        Args:
            noise_prob: Probability of adding noise
            noise_std: Standard deviation of noise
        """
        super().__init__()
        self.noise_prob = noise_prob
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise.

        Args:
            x: Input tensor [batch_size, seq_len, num_channels]

        Returns:
            Noisy tensor [batch_size, seq_len, num_channels]
        """
        if not self.training or random.random() > self.noise_prob:
            return x

        noise = torch.randn_like(x) * self.noise_std
        return x + noise


class SignalScaling(nn.Module):
    """
    Randomly scale neural signals to simulate amplitude variations.
    """

    def __init__(
        self,
        scale_prob: float = 0.1,
        scale_range: Tuple[float, float] = (0.8, 1.2)
    ):
        """
        Initialize signal scaling.

        Args:
            scale_prob: Probability of applying scaling
            scale_range: Range of scaling factors (min, max)
        """
        super().__init__()
        self.scale_prob = scale_prob
        self.scale_range = scale_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply signal scaling.

        Args:
            x: Input tensor [batch_size, seq_len, num_channels]

        Returns:
            Scaled tensor [batch_size, seq_len, num_channels]
        """
        if not self.training or random.random() > self.scale_prob:
            return x

        batch_size = x.shape[0]
        x_scaled = x.clone()

        for b in range(batch_size):
            scale_factor = random.uniform(*self.scale_range)
            x_scaled[b] = x[b] * scale_factor

        return x_scaled


class PhonemeAugmentation(nn.Module):
    """
    Augmentations for phoneme sequences (labels).
    """

    def __init__(
        self,
        insertion_prob: float = 0.05,
        deletion_prob: float = 0.05,
        substitution_prob: float = 0.05,
        vocab_size: int = 40,
        blank_idx: int = 0
    ):
        """
        Initialize phoneme augmentation.

        Args:
            insertion_prob: Probability of inserting phonemes
            deletion_prob: Probability of deleting phonemes
            substitution_prob: Probability of substituting phonemes
            vocab_size: Size of phoneme vocabulary
            blank_idx: Index of blank token
        """
        super().__init__()
        self.insertion_prob = insertion_prob
        self.deletion_prob = deletion_prob
        self.substitution_prob = substitution_prob
        self.vocab_size = vocab_size
        self.blank_idx = blank_idx

    def forward(
        self,
        phonemes: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply phoneme augmentation.

        Args:
            phonemes: Phoneme sequences [batch_size, max_len]
            lengths: Actual sequence lengths [batch_size]

        Returns:
            Augmented phonemes and lengths
        """
        if not self.training:
            return phonemes, lengths

        batch_size = phonemes.shape[0]
        augmented_phonemes = []
        augmented_lengths = []

        for b in range(batch_size):
            seq = phonemes[b, :lengths[b]].tolist()

            # Apply augmentations
            if random.random() < self.insertion_prob:
                seq = self._insert_phonemes(seq)

            if random.random() < self.deletion_prob:
                seq = self._delete_phonemes(seq)

            if random.random() < self.substitution_prob:
                seq = self._substitute_phonemes(seq)

            augmented_phonemes.append(seq)
            augmented_lengths.append(len(seq))

        # Pad sequences
        max_len = max(augmented_lengths)
        padded_phonemes = torch.zeros(batch_size, max_len, dtype=phonemes.dtype, device=phonemes.device)

        for b, seq in enumerate(augmented_phonemes):
            padded_phonemes[b, :len(seq)] = torch.tensor(seq, device=phonemes.device)

        return padded_phonemes, torch.tensor(augmented_lengths, device=lengths.device)

    def _insert_phonemes(self, seq: List[int]) -> List[int]:
        """Insert random phonemes"""
        if len(seq) == 0:
            return seq

        insert_pos = random.randint(0, len(seq))
        insert_phoneme = random.randint(1, self.vocab_size - 1)  # Avoid blank
        seq.insert(insert_pos, insert_phoneme)
        return seq

    def _delete_phonemes(self, seq: List[int]) -> List[int]:
        """Delete random phonemes"""
        if len(seq) <= 1:
            return seq

        delete_pos = random.randint(0, len(seq) - 1)
        del seq[delete_pos]
        return seq

    def _substitute_phonemes(self, seq: List[int]) -> List[int]:
        """Substitute random phonemes"""
        if len(seq) == 0:
            return seq

        sub_pos = random.randint(0, len(seq) - 1)
        sub_phoneme = random.randint(1, self.vocab_size - 1)  # Avoid blank
        seq[sub_pos] = sub_phoneme
        return seq


class ComposeTransforms(nn.Module):
    """
    Compose multiple transforms together.
    """

    def __init__(self, transforms: List[nn.Module]):
        """
        Initialize composed transforms.

        Args:
            transforms: List of transform modules
        """
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all transforms sequentially.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        for transform in self.transforms:
            x = transform(x)
        return x


class NeuralAugmentationPipeline(nn.Module):
    """
    Complete augmentation pipeline for neural signals.
    """

    def __init__(
        self,
        temporal_mask_prob: float = 0.1,
        time_warp_prob: float = 0.1,
        channel_drop_prob: float = 0.1,
        noise_prob: float = 0.1,
        scale_prob: float = 0.1,
        max_mask_length: int = 10,
        max_channels_to_drop: int = 50,
        noise_std: float = 0.1,
        scale_range: Tuple[float, float] = (0.8, 1.2)
    ):
        """
        Initialize augmentation pipeline.

        Args:
            Various augmentation parameters
        """
        super().__init__()

        self.pipeline = ComposeTransforms([
            TemporalMasking(
                mask_prob=temporal_mask_prob,
                max_mask_length=max_mask_length
            ),
            TimeWarping(
                warp_prob=time_warp_prob
            ),
            ChannelDropout(
                drop_prob=channel_drop_prob,
                max_channels_to_drop=max_channels_to_drop
            ),
            GaussianNoise(
                noise_prob=noise_prob,
                noise_std=noise_std
            ),
            SignalScaling(
                scale_prob=scale_prob,
                scale_range=scale_range
            )
        ])

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentation pipeline to batch.

        Args:
            batch: Dictionary containing 'neural_features' and other keys

        Returns:
            Augmented batch
        """
        augmented_batch = batch.copy()

        if 'neural_features' in batch:
            augmented_batch['neural_features'] = self.pipeline(batch['neural_features'])

        return augmented_batch


def create_augmentation_pipeline(config: Dict) -> NeuralAugmentationPipeline:
    """
    Create augmentation pipeline from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured augmentation pipeline
    """
    return NeuralAugmentationPipeline(
        temporal_mask_prob=config.get('temporal_mask_prob', 0.1),
        time_warp_prob=config.get('time_warp_prob', 0.1),
        channel_drop_prob=config.get('channel_drop_prob', 0.1),
        noise_prob=config.get('noise_prob', 0.1),
        scale_prob=config.get('scale_prob', 0.1),
        max_mask_length=config.get('max_mask_length', 10),
        max_channels_to_drop=config.get('max_channels_to_drop', 50),
        noise_std=config.get('noise_std', 0.1),
        scale_range=config.get('scale_range', (0.8, 1.2))
    )