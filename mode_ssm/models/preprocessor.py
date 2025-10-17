"""
Neural signal preprocessor for MODE-SSM.
Handles normalization, channel gating, and temporal augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ChannelAttention(nn.Module):
    """Channel attention mechanism for neural signal preprocessing"""

    def __init__(self, num_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.num_channels = num_channels
        self.reduction_ratio = reduction_ratio

        # Global pooling and channel attention
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_gate = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.

        Args:
            x: Input tensor [batch_size, seq_len, num_channels]

        Returns:
            Attention-weighted tensor [batch_size, seq_len, num_channels]
        """
        # x: [B, T, C]
        x_permuted = x.permute(0, 2, 1)  # [B, C, T]

        # Global average pooling across time dimension
        pooled = self.global_pool(x_permuted)  # [B, C, 1]
        pooled = pooled.squeeze(-1)  # [B, C]

        # Channel attention weights
        attention_weights = self.channel_gate(pooled)  # [B, C]
        attention_weights = attention_weights.unsqueeze(1)  # [B, 1, C]

        # Apply attention
        attended = x * attention_weights  # [B, T, C]

        return attended


class NeuralPreprocessor(nn.Module):
    """
    Neural signal preprocessor for brain-to-text decoding.

    Features:
    - Batch normalization with running statistics
    - Optional channel attention for electrode importance
    - Temporal convolution for local context
    - Projection to model dimension
    - Dropout for regularization
    """

    def __init__(
        self,
        num_channels: int,
        d_model: int,
        normalization_momentum: float = 0.1,
        channel_attention: bool = True,
        conv_kernel_size: int = 7,
        dropout: float = 0.1,
        attention_reduction_ratio: int = 16
    ):
        """
        Initialize neural preprocessor.

        Args:
            num_channels: Number of input channels (typically 512 for T15)
            d_model: Model dimension for output
            normalization_momentum: Momentum for batch normalization
            channel_attention: Whether to use channel attention
            conv_kernel_size: Kernel size for temporal convolution
            dropout: Dropout probability
            attention_reduction_ratio: Reduction ratio for channel attention
        """
        super().__init__()

        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.num_channels = num_channels
        self.d_model = d_model
        self.channel_attention = channel_attention

        # Batch normalization for neural signal conditioning
        # This is crucial for neural data which can have varying scales
        self.normalization = nn.BatchNorm1d(
            num_channels,
            momentum=normalization_momentum,
            track_running_stats=True
        )

        # Channel attention for electrode importance weighting
        if channel_attention:
            self.channel_gate = ChannelAttention(
                num_channels,
                reduction_ratio=attention_reduction_ratio
            )
        else:
            self.channel_gate = None

        # Temporal convolution for local context modeling
        # This captures short-term dependencies in neural signals
        padding = conv_kernel_size // 2  # Same padding
        self.conv1d = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=conv_kernel_size,
            padding=padding,
            groups=num_channels  # Depthwise convolution for efficiency
        )

        # Projection to model dimension
        self.projection = nn.Linear(num_channels, d_model)

        # Layer normalization after projection
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        # Initialize conv1d weights
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

        # Initialize projection layer
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through neural preprocessor.

        Args:
            x: Input neural features [batch_size, seq_len, num_channels]

        Returns:
            Preprocessed features [batch_size, seq_len, d_model]
        """
        # Input shape: [B, T, C]
        batch_size, seq_len, num_channels = x.shape

        if num_channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {num_channels}"
            )

        # Batch normalization requires [B, C, T] format
        x_bn = x.permute(0, 2, 1)  # [B, C, T]
        x_bn = self.normalization(x_bn)
        x_bn = x_bn.permute(0, 2, 1)  # [B, T, C]

        # Apply channel attention if enabled
        if self.channel_gate is not None:
            x_bn = self.channel_gate(x_bn)

        # Temporal convolution for local context
        # Conv1d expects [B, C, T]
        x_conv = x_bn.permute(0, 2, 1)  # [B, C, T]
        x_conv = self.conv1d(x_conv)
        x_conv = F.relu(x_conv)  # Non-linearity after convolution
        x_conv = x_conv.permute(0, 2, 1)  # [B, T, C]

        # Residual connection
        x_residual = x_bn + x_conv

        # Project to model dimension
        x_proj = self.projection(x_residual)  # [B, T, d_model]

        # Layer normalization
        x_norm = self.layer_norm(x_proj)

        # Dropout
        x_out = self.dropout(x_norm)

        return x_out

    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.d_model

    def reset_running_stats(self):
        """Reset running statistics for batch normalization"""
        self.normalization.reset_running_stats()

    def get_running_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current running statistics"""
        return (
            self.normalization.running_mean.clone(),
            self.normalization.running_var.clone()
        )

    def set_running_stats(
        self,
        running_mean: torch.Tensor,
        running_var: torch.Tensor
    ):
        """Set running statistics"""
        self.normalization.running_mean.copy_(running_mean)
        self.normalization.running_var.copy_(running_var)

    def extra_repr(self) -> str:
        """Extra representation string"""
        return (
            f'num_channels={self.num_channels}, '
            f'd_model={self.d_model}, '
            f'channel_attention={self.channel_attention}'
        )