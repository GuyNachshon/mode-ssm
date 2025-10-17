"""
Bidirectional Mamba SSM encoder for MODE-SSM.
Implements efficient state-space modeling for neural sequence processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Install with 'pip install mamba-ssm' for full functionality.")


class MambaBlock(nn.Module):
    """Single Mamba block with layer normalization and residual connection"""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        # Pre-normalization
        self.norm = nn.LayerNorm(d_model)

        # Mamba layer
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                conv_bias=conv_bias,
                bias=bias,
                use_fast_path=use_fast_path,
            )
        else:
            # Fallback: Use standard LSTM when mamba-ssm not available
            # This provides basic sequence modeling capabilities
            self.mamba = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                batch_first=True,
                bidirectional=False
            )
            self.is_fallback = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba block.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Pre-normalization
        x_norm = self.norm(x)

        # Mamba processing
        if MAMBA_AVAILABLE:
            y = self.mamba(x_norm)
        else:
            # LSTM fallback
            y, _ = self.mamba(x_norm)

        # Residual connection
        return x + y


class BidirectionalMamba(nn.Module):
    """Bidirectional Mamba layer for capturing both past and future context"""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model

        # Forward and backward Mamba blocks
        self.forward_mamba = MambaBlock(
            d_model=d_model // 2,  # Split dimensions for bidirectional
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )

        self.backward_mamba = MambaBlock(
            d_model=d_model // 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )

        # Input projection to split dimensions
        self.input_proj = nn.Linear(d_model, d_model)

        # Output projection to combine bidirectional outputs
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Project input
        x_proj = self.input_proj(x)

        # Split for bidirectional processing
        x_forward = x_proj[..., :self.d_model // 2]
        x_backward = x_proj[..., self.d_model // 2:]

        # Forward pass
        y_forward = self.forward_mamba(x_forward)

        # Backward pass (reverse sequence)
        x_backward_rev = torch.flip(x_backward, dims=[1])
        y_backward_rev = self.backward_mamba(x_backward_rev)
        y_backward = torch.flip(y_backward_rev, dims=[1])

        # Concatenate bidirectional outputs
        y_concat = torch.cat([y_forward, y_backward], dim=-1)

        # Final projection
        output = self.output_proj(y_concat)

        return output


class MambaEncoder(nn.Module):
    """
    Multi-layer Mamba encoder for neural sequence modeling.

    Features:
    - Bidirectional state-space modeling
    - Residual connections and layer normalization
    - Gradient checkpointing support
    - Efficient long sequence processing
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 8,
        bidirectional: bool = True,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        gradient_checkpointing: bool = False,
        **kwargs
    ):
        """
        Initialize Mamba encoder.

        Args:
            d_model: Model dimension
            d_state: State dimension for SSM
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
            n_layers: Number of Mamba layers
            bidirectional: Whether to use bidirectional processing
            dropout: Dropout probability
            layer_norm_eps: Layer normalization epsilon
            gradient_checkpointing: Whether to use gradient checkpointing
            **kwargs: Additional arguments for Mamba blocks
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_state <= 0:
            raise ValueError(f"d_state must be positive, got {d_state}")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")

        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.gradient_checkpointing = gradient_checkpointing

        # Create Mamba layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if bidirectional:
                layer = BidirectionalMamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **kwargs
                )
            else:
                layer = MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **kwargs
                )
            self.layers.append(layer)

        # Dropout between layers
        self.dropout = nn.Dropout(dropout)

        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Mamba encoder.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len]

        Returns:
            Encoded tensor [batch_size, seq_len, d_model]
        """
        # Process through Mamba layers
        hidden_states = x

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states)

            # Apply dropout between layers
            if i < len(self.layers) - 1:  # Don't apply dropout after last layer
                hidden_states = self.dropout(hidden_states)

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match hidden dimension
            mask = mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states * mask

        return hidden_states

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_flops_per_token(self) -> int:
        """Estimate FLOPs per token (approximate)"""
        # This is an approximation for Mamba
        # Actual FLOPs depend on implementation details
        flops_per_layer = 6 * self.d_model * self.d_model  # Rough estimate
        return self.n_layers * flops_per_layer

    def extra_repr(self) -> str:
        """Extra representation string"""
        return (
            f'd_model={self.d_model}, '
            f'd_state={self.d_state}, '
            f'n_layers={self.n_layers}, '
            f'bidirectional={self.bidirectional}'
        )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (optional for Mamba)"""

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor"""
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)


class MambaEncoderWithPE(MambaEncoder):
    """Mamba encoder with optional positional encoding"""

    def __init__(
        self,
        d_model: int,
        use_positional_encoding: bool = False,
        max_seq_len: int = 10000,
        **kwargs
    ):
        super().__init__(d_model=d_model, **kwargs)

        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        else:
            self.pos_encoding = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional positional encoding"""
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        return super().forward(x, mask)