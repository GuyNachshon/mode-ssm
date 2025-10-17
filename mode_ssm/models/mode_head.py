"""
Speaking mode classification head for MODE-SSM.
Classifies neural signals as silent vs vocalized speech with contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Any


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence-to-vector conversion"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Attention mechanism
        self.attention_proj = nn.Linear(d_model, d_model)
        self.attention_weights = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention pooling to sequence.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional mask [batch_size, seq_len]

        Returns:
            Pooled tensor [batch_size, d_model]
        """
        # Project for attention computation
        projected = torch.tanh(self.attention_proj(x))  # [B, T, d_model]
        attention_scores = self.attention_weights(projected).squeeze(-1)  # [B, T]

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, T]
        attention_weights = self.dropout(attention_weights)

        # Apply attention to input
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # [B, d_model]

        return pooled


class ModeClassificationHead(nn.Module):
    """
    Mode classification head for identifying speaking modes.

    Features:
    - Binary classification (silent vs vocalized)
    - Multiple pooling strategies (global_avg, global_max, attention_pool)
    - Optional contrastive learning features
    - Dropout regularization
    """

    def __init__(
        self,
        d_model: int,
        num_modes: int = 2,
        dropout: float = 0.1,
        contrastive_learning: bool = True,
        pooling_type: str = "global_avg",
        contrastive_dim: int = 128,
        temperature: float = 0.07
    ):
        """
        Initialize mode classification head.

        Args:
            d_model: Input feature dimension
            num_modes: Number of mode classes (typically 2 for silent/vocalized)
            dropout: Dropout probability
            contrastive_learning: Whether to enable contrastive learning features
            pooling_type: Type of pooling ("global_avg", "global_max", "attention_pool")
            contrastive_dim: Dimension for contrastive features
            temperature: Temperature for contrastive learning
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_modes < 2:
            raise ValueError(f"num_modes must be at least 2, got {num_modes}")

        self.d_model = d_model
        self.num_modes = num_modes
        self.contrastive_learning = contrastive_learning
        self.pooling_type = pooling_type
        self.contrastive_dim = contrastive_dim
        self.temperature = temperature

        # Pooling strategy
        if pooling_type == "global_avg":
            self.pooling = self._global_avg_pool
        elif pooling_type == "global_max":
            self.pooling = self._global_max_pool
        elif pooling_type == "attention_pool":
            self.pooling = AttentionPooling(d_model, dropout)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        # Feature processing layers
        self.feature_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Linear(d_model // 4, num_modes)

        # Contrastive learning projection
        if contrastive_learning:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(d_model // 4, contrastive_dim),
                nn.ReLU(inplace=True),
                nn.Linear(contrastive_dim, contrastive_dim),
            )
        else:
            self.contrastive_proj = None

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _global_avg_pool(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Global average pooling"""
        if mask is not None:
            # Masked average
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            masked_x = x * mask_expanded
            pooled = masked_x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return pooled

    def _global_max_pool(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Global max pooling"""
        if mask is not None:
            # Masked max pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            masked_x = x.masked_fill(~mask_expanded, float('-inf'))
            pooled = masked_x.max(dim=1)[0]
            # Handle case where all values are masked
            pooled = pooled.masked_fill(pooled.isinf(), 0.0)
        else:
            pooled = x.max(dim=1)[0]
        return pooled

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through mode classification head.

        Args:
            x: Input features [batch_size, seq_len, d_model]
            mask: Optional sequence mask [batch_size, seq_len]
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
            - mode_logits: Classification logits [batch_size, num_modes]
            - contrastive_features: Contrastive features if enabled [batch_size, contrastive_dim]
            - pooled_features: Pooled features if return_features=True [batch_size, d_model//4]
        """
        batch_size, seq_len, d_model = x.shape

        # Apply pooling to convert sequence to single vector
        if callable(self.pooling):
            # Attention pooling
            pooled = self.pooling(x, mask)
        else:
            # Function-based pooling
            pooled = self.pooling(x, mask)

        # Feature processing
        features = self.feature_layers(pooled)  # [B, d_model//4]

        # Classification logits
        mode_logits = self.classifier(features)  # [B, num_modes]

        # Prepare outputs
        outputs = {'mode_logits': mode_logits}

        # Contrastive features
        if self.contrastive_proj is not None:
            contrastive_features = self.contrastive_proj(features)  # [B, contrastive_dim]
            # L2 normalize for contrastive learning
            contrastive_features = F.normalize(contrastive_features, p=2, dim=-1)
            outputs['contrastive_features'] = contrastive_features

        # Optional feature return
        if return_features:
            outputs['pooled_features'] = features

        return outputs

    def compute_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: Contrastive features [batch_size, contrastive_dim]
            labels: Mode labels [batch_size] (optional, returns 0 if None)
            mask: Optional mask for valid samples [batch_size]

        Returns:
            Contrastive loss tensor
        """
        if not self.contrastive_learning:
            raise ValueError("Contrastive learning not enabled")

        device = features.device

        # Return zero loss if labels not provided
        if labels is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        batch_size = features.shape[0]

        # Apply mask if provided
        if mask is not None:
            features = features[mask]
            labels = labels[mask]
            batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create label mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)

        # Remove diagonal elements (self-similarity)
        mask_no_diag = torch.ones_like(mask_positive) - torch.eye(batch_size, device=device)
        mask_positive = mask_positive * mask_no_diag

        # Compute contrastive loss
        # For each sample, find positive and negative pairs
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log probabilities
        exp_logits = torch.exp(logits) * mask_no_diag
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask_positive * log_prob).sum(1) / mask_positive.sum(1).clamp(min=1)

        # Contrastive loss (negative log-likelihood)
        loss = -mean_log_prob_pos.mean()

        return loss

    def get_mode_probabilities(self, mode_logits: torch.Tensor) -> torch.Tensor:
        """Convert mode logits to probabilities"""
        return F.softmax(mode_logits, dim=-1)

    def get_mode_predictions(self, mode_logits: torch.Tensor) -> torch.Tensor:
        """Get mode predictions from logits"""
        return torch.argmax(mode_logits, dim=-1)

    def get_confidence_scores(self, mode_logits: torch.Tensor) -> torch.Tensor:
        """Get confidence scores (max probability)"""
        probs = self.get_mode_probabilities(mode_logits)
        return torch.max(probs, dim=-1)[0]

    def extra_repr(self) -> str:
        """Extra representation string"""
        return (
            f'd_model={self.d_model}, '
            f'num_modes={self.num_modes}, '
            f'pooling_type={self.pooling_type}, '
            f'contrastive_learning={self.contrastive_learning}'
        )