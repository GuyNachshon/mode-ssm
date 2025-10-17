"""
Flow Bridge Denoiser for MODE-SSM.
Implements diffusion/flow-based denoising for neural signals using modern generative modeling.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class FlowBridgeConfig:
    """Configuration for flow bridge denoiser"""

    # Model architecture
    d_model: int = 512
    num_channels: int = 512
    num_layers: int = 6
    num_heads: int = 8
    feedforward_dim: int = 2048
    dropout: float = 0.1

    # Flow bridge parameters
    num_flow_steps: int = 20
    noise_schedule: str = "cosine"  # linear, cosine, sigmoid
    parameterization: str = "v"  # eps, x0, v (velocity parameterization)

    # Denoising parameters
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0  # Schedule parameter

    # Training parameters
    loss_weighting: str = "snr"  # uniform, snr, snr_trunc
    min_snr_gamma: float = 5.0

    # Inference parameters
    num_inference_steps: int = 20
    guidance_scale: float = 1.0
    eta: float = 0.0  # DDIM eta parameter


class TimestepEmbedding(nn.Module):
    """Timestep embedding for diffusion models"""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # Learnable embedding layers
        self.time_embed = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            timesteps: 1-D tensor of N indices, one per batch element

        Returns:
            Embedding tensor [N, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2

        # Create frequency embeddings
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim
        )

        # Apply to timesteps
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        # Apply learnable transformation
        embedding = self.time_embed(embedding)

        return embedding


class ResidualBlock(nn.Module):
    """Residual block for flow bridge network"""

    def __init__(self, dim: int, time_dim: int, dropout: float = 0.1):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim)
        )

        self.block = nn.Sequential(
            nn.GroupNorm(32, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.Dropout(dropout),
            nn.GroupNorm(32, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.Dropout(dropout),
        )

        self.residual_projection = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, seq_len]
            time_emb: [batch_size, time_dim]

        Returns:
            Output tensor [batch_size, channels, seq_len]
        """
        h = x

        # Apply time embedding
        time_out = self.time_mlp(time_emb)
        h = h + time_out[..., None]  # Broadcast over seq_len

        # Apply residual block
        h = self.block(h)

        return self.residual_projection(x) + h


class AttentionBlock(nn.Module):
    """Self-attention block for denoiser"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv1d(dim, dim * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, seq_len]

        Returns:
            Output tensor [batch_size, channels, seq_len]
        """
        batch_size, channels, seq_len = x.shape
        residual = x

        # Normalize
        x = self.norm(x)

        # Get Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.view(batch_size, self.num_heads, self.head_dim, seq_len)
        k = k.view(batch_size, self.num_heads, self.head_dim, seq_len)
        v = v.view(batch_size, self.num_heads, self.head_dim, seq_len)

        # Compute attention
        scores = torch.einsum('bhdk,bhdl->bhkl', q, k) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.einsum('bhkl,bhdl->bhdk', attn, v)
        out = out.contiguous().view(batch_size, channels, seq_len)

        # Output projection
        out = self.proj_out(out)

        return residual + out


class FlowBridgeNetwork(nn.Module):
    """Neural network for flow bridge denoising"""

    def __init__(self, config: FlowBridgeConfig):
        super().__init__()
        self.config = config

        # Time embedding
        time_dim = config.d_model * 4
        self.time_embed = TimestepEmbedding(config.d_model, max_period=10000)

        # Input projection
        self.input_proj = nn.Conv1d(config.num_channels, config.d_model, kernel_size=1)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock(config.d_model, time_dim, config.dropout)
            for _ in range(config.num_layers // 2)
        ])

        # Middle attention block
        self.middle_attn = AttentionBlock(config.d_model, config.num_heads, config.dropout)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(config.d_model, time_dim, config.dropout)
            for _ in range(config.num_layers // 2)
        ])

        # Output projection
        self.output_proj = nn.Conv1d(config.d_model, config.num_channels, kernel_size=1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through denoising network.

        Args:
            x: Noisy input [batch_size, seq_len, num_channels]
            timesteps: Timestep tensor [batch_size]
            condition: Optional conditioning [batch_size, seq_len, condition_dim]

        Returns:
            Denoised output [batch_size, seq_len, num_channels]
        """
        # Convert to channel-first format
        x = x.transpose(1, 2)  # [batch_size, num_channels, seq_len]

        # Time embedding
        time_emb = self.time_embed(timesteps)

        # Input projection
        h = self.input_proj(x)

        # Store skip connections
        skip_connections = []

        # Encoder
        for block in self.encoder_blocks:
            h = block(h, time_emb)
            skip_connections.append(h)

        # Middle attention
        h = self.middle_attn(h)

        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            if i < len(skip_connections):
                h = h + skip_connections[-(i+1)]
            h = block(h, time_emb)

        # Output projection
        output = self.output_proj(h)

        # Convert back to seq-first format
        output = output.transpose(1, 2)  # [batch_size, seq_len, num_channels]

        return output


class NoiseScheduler:
    """Noise scheduler for diffusion process"""

    def __init__(self, config: FlowBridgeConfig):
        self.config = config
        self.num_train_timesteps = config.num_flow_steps

        # Create noise schedule
        if config.noise_schedule == "linear":
            self.betas = torch.linspace(0.0001, 0.02, config.num_flow_steps)
        elif config.noise_schedule == "cosine":
            self.betas = self._cosine_schedule(config.num_flow_steps)
        elif config.noise_schedule == "sigmoid":
            self.betas = self._sigmoid_schedule(config.num_flow_steps)
        else:
            raise ValueError(f"Unknown noise schedule: {config.noise_schedule}")

        # Compute schedule parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For v-parameterization
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()

    def _cosine_schedule(self, timesteps: int, s: float = 0.008):
        """Cosine noise schedule"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _sigmoid_schedule(self, timesteps: int):
        """Sigmoid noise schedule"""
        betas = torch.linspace(-6, 6, timesteps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        return betas

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to samples"""
        # Make sure alphas_cumprod are on the same device
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Get velocity for v-parameterization"""
        # Make sure alphas_cumprod are on the same device
        alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


class FlowBridgeDenoiser(nn.Module):
    """
    Complete flow bridge denoiser for neural signals.
    Implements diffusion-based denoising with configurable schedules and parameterizations.
    """

    def __init__(self, config: FlowBridgeConfig):
        super().__init__()
        self.config = config

        # Core components
        self.network = FlowBridgeNetwork(config)
        self.scheduler = NoiseScheduler(config)

        # Loss weighting
        self.loss_weighting = config.loss_weighting

    def forward(
        self,
        clean_samples: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            clean_samples: Clean neural data [batch_size, seq_len, num_channels]
            condition: Optional conditioning information

        Returns:
            Dictionary with loss and predictions
        """
        batch_size = clean_samples.shape[0]
        device = clean_samples.device

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.config.num_flow_steps,
            (batch_size,), device=device, dtype=torch.long
        )

        # Sample noise
        noise = torch.randn_like(clean_samples)

        # Add noise to clean samples
        noisy_samples = self.scheduler.add_noise(clean_samples, noise, timesteps)

        # Get target based on parameterization
        if self.config.parameterization == "eps":
            target = noise
        elif self.config.parameterization == "x0":
            target = clean_samples
        elif self.config.parameterization == "v":
            target = self.scheduler.get_velocity(clean_samples, noise, timesteps)
        else:
            raise ValueError(f"Unknown parameterization: {self.config.parameterization}")

        # Predict with network
        model_pred = self.network(noisy_samples, timesteps, condition)

        # Compute loss
        loss = F.mse_loss(model_pred, target, reduction="none")

        # Apply loss weighting
        if self.loss_weighting == "snr":
            # Signal-to-noise ratio weighting
            alphas_cumprod = self.scheduler.alphas_cumprod.to(device)[timesteps]
            snr = alphas_cumprod / (1 - alphas_cumprod)
            loss_weights = torch.minimum(snr, torch.ones_like(snr) * self.config.min_snr_gamma)
            loss = loss * loss_weights.view(-1, 1, 1)

        # Reduce loss
        loss = loss.mean()

        return {
            'loss': loss,
            'model_pred': model_pred,
            'target': target,
            'noisy_samples': noisy_samples
        }

    @torch.no_grad()
    def denoise(
        self,
        noisy_samples: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        condition: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Denoise samples using DDIM sampling.

        Args:
            noisy_samples: Noisy input [batch_size, seq_len, num_channels]
            num_inference_steps: Number of denoising steps
            condition: Optional conditioning
            generator: Random generator for reproducibility

        Returns:
            Denoised samples [batch_size, seq_len, num_channels]
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps

        device = noisy_samples.device

        # Create inference schedule
        step_size = self.config.num_flow_steps // num_inference_steps
        timesteps = torch.arange(self.config.num_flow_steps - 1, -1, -step_size, device=device)

        # Start with noisy samples
        sample = noisy_samples.clone()

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand timestep to batch dimension
            timestep = t.expand(sample.shape[0])

            # Predict noise/velocity/x0
            model_output = self.network(sample, timestep, condition)

            # Convert to noise prediction if needed
            if self.config.parameterization == "v":
                # Convert velocity to noise
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                pred_noise = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
            elif self.config.parameterization == "x0":
                # Convert x0 to noise
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                pred_noise = (sample - (alpha_prod_t ** 0.5) * model_output) / (beta_prod_t ** 0.5)
            else:
                pred_noise = model_output

            # DDIM step
            if i < len(timesteps) - 1:
                next_t = timesteps[i + 1]
                sample = self._ddim_step(sample, pred_noise, t, next_t)
            else:
                # Final step
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                sample = (sample - (1 - alpha_prod_t) ** 0.5 * pred_noise) / (alpha_prod_t ** 0.5)

        return sample

    def _ddim_step(
        self,
        sample: torch.Tensor,
        pred_noise: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor
    ) -> torch.Tensor:
        """Single DDIM denoising step"""
        device = sample.device

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].to(device)
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep].to(device)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_next = 1 - alpha_prod_t_next

        # Predicted x0
        pred_original_sample = (sample - beta_prod_t ** 0.5 * pred_noise) / alpha_prod_t ** 0.5

        # Direction pointing to x_t
        pred_sample_direction = (1 - alpha_prod_t_next - self.config.eta ** 2 * (beta_prod_t_next - beta_prod_t)) ** 0.5 * pred_noise

        # x_{t-1}
        prev_sample = alpha_prod_t_next ** 0.5 * pred_original_sample + pred_sample_direction

        if self.config.eta > 0:
            # Add noise
            variance = (beta_prod_t_next - beta_prod_t) * self.config.eta ** 2
            if variance > 0:
                noise = torch.randn_like(sample)
                prev_sample = prev_sample + variance ** 0.5 * noise

        return prev_sample

    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)


def create_flow_bridge_denoiser(config: Union[FlowBridgeConfig, DictConfig]) -> FlowBridgeDenoiser:
    """Factory function to create flow bridge denoiser"""
    if isinstance(config, DictConfig):
        # Convert OmegaConf to dataclass
        config = FlowBridgeConfig(**config)

    return FlowBridgeDenoiser(config)