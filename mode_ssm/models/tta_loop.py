"""
Test-Time Adaptation (TTA) loop for MODE-SSM.
Implements session-level statistics adaptation and entropy minimization
to handle neural drift during inference.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class TTAConfig:
    """Configuration for test-time adaptation"""

    # Session adaptation settings
    session_adaptation_enabled: bool = True
    adaptation_lr: float = 0.001
    adaptation_steps: int = 5
    statistics_momentum: float = 0.9
    min_samples_for_adaptation: int = 10
    adaptation_layers: List[str] = field(default_factory=lambda: ['preprocessor'])
    max_adaptation_steps_per_session: int = 50
    adaptation_warmup_samples: int = 5

    # Entropy minimization settings
    entropy_minimization_enabled: bool = True
    entropy_threshold: float = 2.0
    entropy_weight: float = 1.0
    confidence_threshold: float = 0.8
    entropy_decay_factor: float = 0.99

    # Memory management
    max_session_history: int = 1000
    cleanup_interval: int = 100

    # Numerical stability
    eps: float = 1e-6
    clip_grad_norm: float = 1.0


@dataclass
class AdaptationStatistics:
    """Statistics for tracking adaptation progress"""

    session_adaptations: Dict[str, int] = field(default_factory=dict)
    entropy_stats: Dict[str, float] = field(default_factory=dict)
    adaptation_step: int = 0
    total_steps: int = 0
    feature_change_magnitude: float = 0.0
    convergence_metrics: Dict[str, float] = field(default_factory=dict)


class SessionStatsAdapter:
    """Handles session-level statistics adaptation for neural drift compensation"""

    def __init__(self, config: TTAConfig):
        self.config = config
        self.session_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.session_sample_counts: Dict[str, int] = defaultdict(int)
        self.adaptation_enabled = config.session_adaptation_enabled

        # Adaptation history for convergence tracking
        self.adaptation_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.max_adaptation_steps_per_session)
        )

    def compute_session_statistics(
        self,
        neural_features: torch.Tensor,
        session_id: str
    ) -> Dict[str, torch.Tensor]:
        """
        Compute session-level statistics from neural features.

        Args:
            neural_features: [batch_size, seq_len, num_channels]
            session_id: Session identifier

        Returns:
            Dictionary with session statistics
        """
        # Compute per-channel statistics across batch and sequence dimensions
        # Shape: [batch_size, seq_len, num_channels] -> [num_channels]

        valid_mask = (neural_features.abs() > self.config.eps)

        # Compute statistics only on valid (non-zero) values
        if valid_mask.any():
            valid_features = neural_features[valid_mask]

            # Reshape to [num_samples, num_channels] for per-channel stats
            reshaped_features = neural_features.view(-1, neural_features.shape[-1])
            valid_samples = (reshaped_features.abs() > self.config.eps).any(dim=1)

            if valid_samples.any():
                features_for_stats = reshaped_features[valid_samples]

                mean_vals = features_for_stats.mean(dim=0)
                var_vals = features_for_stats.var(dim=0, unbiased=False)

                # Add small epsilon for numerical stability
                var_vals = var_vals + self.config.eps
            else:
                # Fallback for edge cases
                mean_vals = torch.zeros(neural_features.shape[-1], device=neural_features.device)
                var_vals = torch.ones(neural_features.shape[-1], device=neural_features.device)
        else:
            # All features are zero - use default stats
            mean_vals = torch.zeros(neural_features.shape[-1], device=neural_features.device)
            var_vals = torch.ones(neural_features.shape[-1], device=neural_features.device)

        return {
            'mean': mean_vals,
            'var': var_vals,
            'count': neural_features.shape[0] * neural_features.shape[1]
        }

    def update_session_statistics(
        self,
        neural_features: torch.Tensor,
        session_id: str
    ) -> Dict[str, torch.Tensor]:
        """
        Update session statistics using exponential moving average.

        Args:
            neural_features: New neural features for the session
            session_id: Session identifier

        Returns:
            Updated session statistics
        """
        new_stats = self.compute_session_statistics(neural_features, session_id)

        if session_id not in self.session_stats:
            # First time seeing this session
            self.session_stats[session_id] = new_stats.copy()
        else:
            # Update using exponential moving average
            momentum = self.config.statistics_momentum
            old_stats = self.session_stats[session_id]

            updated_stats = {
                'mean': momentum * old_stats['mean'] + (1 - momentum) * new_stats['mean'],
                'var': momentum * old_stats['var'] + (1 - momentum) * new_stats['var'],
                'count': old_stats['count'] + new_stats['count']
            }

            self.session_stats[session_id] = updated_stats

        self.session_sample_counts[session_id] += neural_features.shape[0]

        # Store adaptation history
        self.adaptation_history[session_id].append({
            'mean_norm': self.session_stats[session_id]['mean'].norm().item(),
            'var_norm': self.session_stats[session_id]['var'].norm().item(),
            'sample_count': self.session_sample_counts[session_id]
        })

        return self.session_stats[session_id]

    def adapt_features(
        self,
        neural_features: torch.Tensor,
        session_ids: List[str]
    ) -> torch.Tensor:
        """
        Adapt neural features based on session statistics.

        Args:
            neural_features: [batch_size, seq_len, num_channels]
            session_ids: List of session IDs for each batch item

        Returns:
            Adapted neural features
        """
        if not self.adaptation_enabled:
            return neural_features

        adapted_features = neural_features.clone()

        for i, session_id in enumerate(session_ids):
            # Check if we have enough data for this session
            if self.session_sample_counts[session_id] < self.config.min_samples_for_adaptation:
                continue

            if session_id not in self.session_stats:
                continue

            session_stats = self.session_stats[session_id]

            # Normalize using session-specific statistics
            features_i = adapted_features[i]  # [seq_len, num_channels]

            # Center using session mean
            centered_features = features_i - session_stats['mean'].unsqueeze(0)

            # Scale using session variance
            std_vals = torch.sqrt(session_stats['var'])
            normalized_features = centered_features / (std_vals.unsqueeze(0) + self.config.eps)

            adapted_features[i] = normalized_features

        return adapted_features

    def get_adaptation_strength(self, session_id: str) -> float:
        """Get adaptation strength for a session based on drift magnitude"""
        if session_id not in self.adaptation_history:
            return 0.0

        history = list(self.adaptation_history[session_id])
        if len(history) < 2:
            return 0.0

        # Compute drift as change in statistics over time
        recent_stats = history[-1]
        early_stats = history[0]

        mean_drift = abs(recent_stats['mean_norm'] - early_stats['mean_norm'])
        var_drift = abs(recent_stats['var_norm'] - early_stats['var_norm'])

        # Normalize by number of samples seen
        sample_ratio = recent_stats['sample_count'] / max(early_stats['sample_count'], 1)

        return (mean_drift + var_drift) * math.log(sample_ratio + 1)

    def cleanup_old_sessions(self):
        """Clean up statistics for old sessions to manage memory"""
        if len(self.session_stats) <= self.config.max_session_history:
            return

        # Remove least recently used sessions
        # For simplicity, remove sessions with lowest sample counts
        session_counts = [(sid, count) for sid, count in self.session_sample_counts.items()]
        session_counts.sort(key=lambda x: x[1])  # Sort by sample count

        sessions_to_remove = [sid for sid, _ in session_counts[:len(session_counts) // 4]]

        for session_id in sessions_to_remove:
            if session_id in self.session_stats:
                del self.session_stats[session_id]
            if session_id in self.session_sample_counts:
                del self.session_sample_counts[session_id]
            if session_id in self.adaptation_history:
                del self.adaptation_history[session_id]


class EntropyMinimizer:
    """Handles entropy minimization for improving prediction confidence"""

    def __init__(self, config: TTAConfig):
        self.config = config
        self.entropy_threshold = config.entropy_threshold
        self.entropy_weight = config.entropy_weight
        self.confidence_threshold = config.confidence_threshold
        self.enabled = config.entropy_minimization_enabled

        # Entropy tracking
        self.entropy_history: deque = deque(maxlen=100)

    def compute_prediction_entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of predictions.

        Args:
            predictions: [batch_size, seq_len, vocab_size] logits

        Returns:
            Per-sample entropy [batch_size]
        """
        # Convert logits to probabilities
        log_probs = F.log_softmax(predictions, dim=-1)
        probs = torch.exp(log_probs)

        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size, seq_len]

        # Average over sequence length
        entropy_per_sample = entropy.mean(dim=1)  # [batch_size]

        return entropy_per_sample

    def identify_high_entropy_samples(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Identify samples with high entropy (low confidence).

        Args:
            predictions: [batch_size, seq_len, vocab_size] logits

        Returns:
            Boolean mask [batch_size] indicating high-entropy samples
        """
        entropy_per_sample = self.compute_prediction_entropy(predictions)
        return entropy_per_sample > self.entropy_threshold

    def compute_entropy_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy loss for minimization.

        Args:
            predictions: [batch_size, seq_len, vocab_size] logits

        Returns:
            Scalar entropy loss
        """
        if not self.enabled:
            return torch.tensor(0.0, device=predictions.device)

        # Convert to probabilities
        log_probs = F.log_softmax(predictions, dim=-1)
        probs = torch.exp(log_probs)

        # Compute entropy
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size, seq_len]

        # Weight by confidence - focus more on uncertain predictions
        max_probs = probs.max(dim=-1)[0]  # [batch_size, seq_len]
        uncertainty_weights = 1.0 - max_probs

        # Weighted entropy loss
        weighted_entropy = entropy * uncertainty_weights

        # Average and scale
        entropy_loss = weighted_entropy.mean() * self.entropy_weight

        # Track entropy for monitoring
        self.entropy_history.append(entropy.mean().item())

        return entropy_loss

    def compute_batch_entropy_statistics(self, predictions: torch.Tensor) -> Dict[str, float]:
        """Compute batch-level entropy statistics for monitoring"""
        entropy_per_sample = self.compute_prediction_entropy(predictions)

        high_entropy_mask = entropy_per_sample > self.entropy_threshold
        high_entropy_fraction = high_entropy_mask.float().mean().item()

        return {
            'mean_entropy': entropy_per_sample.mean().item(),
            'std_entropy': entropy_per_sample.std().item(),
            'min_entropy': entropy_per_sample.min().item(),
            'max_entropy': entropy_per_sample.max().item(),
            'high_entropy_fraction': high_entropy_fraction
        }


class TTALoop:
    """Complete test-time adaptation loop combining session adaptation and entropy minimization"""

    def __init__(self, config: TTAConfig, model: nn.Module):
        self.config = config
        self.model = model

        # Initialize adaptation components
        self.session_adapter = SessionStatsAdapter(config)
        self.entropy_minimizer = EntropyMinimizer(config)

        # Optimization setup for TTA
        self.adaptation_optimizer = None
        self._setup_adaptation_optimizer()

        # Tracking
        self.adaptation_step_count = 0
        self.cleanup_counter = 0

    def _setup_adaptation_optimizer(self):
        """Setup optimizer for adaptation parameters"""
        # Get adaptable parameters based on configuration
        adaptable_params = []

        for layer_name in self.config.adaptation_layers:
            if layer_name == 'preprocessor' and hasattr(self.model, 'preprocessor'):
                # Add preprocessing layer parameters
                adaptable_params.extend(self.model.preprocessor.parameters())
            elif layer_name == 'encoder' and hasattr(self.model, 'encoder'):
                # Add encoder normalization parameters (but not main weights)
                for name, param in self.model.encoder.named_parameters():
                    if 'norm' in name.lower() or 'bn' in name.lower():
                        adaptable_params.append(param)

        if adaptable_params:
            self.adaptation_optimizer = torch.optim.Adam(
                adaptable_params,
                lr=self.config.adaptation_lr
            )

    def adapt_single_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single adaptation step.

        Args:
            batch: Input batch with neural_features, sequence_lengths, session_ids

        Returns:
            Adapted batch with additional statistics
        """
        neural_features = batch['neural_features']
        sequence_lengths = batch['sequence_lengths']
        session_ids = batch['session_ids']

        # 1. Update session statistics
        for i, session_id in enumerate(session_ids):
            self.session_adapter.update_session_statistics(
                neural_features[i:i+1], session_id
            )

        # 2. Adapt features using session statistics
        adapted_features = self.session_adapter.adapt_features(
            neural_features, session_ids
        )

        # 3. Run model to get predictions for entropy minimization
        if self.config.entropy_minimization_enabled and self.adaptation_optimizer is not None:
            # Enable gradients for adaptation
            adapted_features.requires_grad_(True)

            # Forward pass
            model_outputs = self.model(
                neural_features=adapted_features,
                sequence_lengths=sequence_lengths
            )

            # Get RNNT predictions for entropy computation
            if 'rnnt_predictions' in model_outputs:
                predictions = model_outputs['rnnt_predictions']

                # Compute entropy loss
                entropy_loss = self.entropy_minimizer.compute_entropy_loss(predictions)

                # Backward pass and adaptation step
                if entropy_loss.item() > 0:
                    entropy_loss.backward(retain_graph=True)

                    # Clip gradients for stability
                    if self.config.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.adaptation_optimizer.param_groups[0]['params'],
                            self.config.clip_grad_norm
                        )

                    self.adaptation_optimizer.step()
                    self.adaptation_optimizer.zero_grad()

        # 4. Compute final adapted features (after parameter updates)
        with torch.no_grad():
            final_adapted_features = self.session_adapter.adapt_features(
                neural_features, session_ids
            )

        # 5. Compute adaptation statistics
        adaptation_stats = self._compute_adaptation_statistics(
            neural_features, final_adapted_features, session_ids
        )

        self.adaptation_step_count += 1

        # 6. Periodic cleanup
        self.cleanup_counter += 1
        if self.cleanup_counter >= self.config.cleanup_interval:
            self.session_adapter.cleanup_old_sessions()
            self.cleanup_counter = 0

        return {
            'neural_features': final_adapted_features.detach(),
            'sequence_lengths': sequence_lengths,
            'session_ids': session_ids,
            'adaptation_stats': adaptation_stats
        }

    def adapt_multi_step(
        self,
        batch: Dict[str, torch.Tensor],
        num_steps: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform multiple adaptation steps.

        Args:
            batch: Input batch
            num_steps: Number of adaptation steps (defaults to config value)

        Returns:
            Adapted batch after multiple steps
        """
        if num_steps is None:
            num_steps = self.config.adaptation_steps

        current_batch = batch
        all_stats = []

        for step in range(num_steps):
            adapted_batch = self.adapt_single_step(current_batch)
            all_stats.append(adapted_batch['adaptation_stats'])

            # Use adapted features for next step
            current_batch = {
                'neural_features': adapted_batch['neural_features'],
                'sequence_lengths': adapted_batch['sequence_lengths'],
                'session_ids': adapted_batch['session_ids']
            }

        # Combine statistics from all steps
        final_stats = self._combine_multi_step_statistics(all_stats)
        final_stats['total_steps'] = num_steps

        return {
            'neural_features': current_batch['neural_features'],
            'sequence_lengths': current_batch['sequence_lengths'],
            'session_ids': current_batch['session_ids'],
            'adaptation_stats': final_stats
        }

    def _compute_adaptation_statistics(
        self,
        original_features: torch.Tensor,
        adapted_features: torch.Tensor,
        session_ids: List[str]
    ) -> Dict[str, Any]:
        """Compute adaptation statistics for monitoring"""

        # Feature change magnitude
        feature_change = torch.norm(adapted_features - original_features).item()

        # Per-session adaptation counts
        session_adaptations = {}
        for session_id in set(session_ids):
            session_adaptations[session_id] = self.session_adapter.session_sample_counts[session_id]

        # Entropy statistics (if available)
        entropy_stats = {}
        if hasattr(self, '_last_predictions'):
            entropy_stats = self.entropy_minimizer.compute_batch_entropy_statistics(
                self._last_predictions
            )

        return {
            'session_adaptations': session_adaptations,
            'entropy_stats': entropy_stats,
            'adaptation_step': self.adaptation_step_count,
            'feature_change_magnitude': feature_change,
            'convergence_metrics': {
                'adaptation_strength': {
                    session_id: self.session_adapter.get_adaptation_strength(session_id)
                    for session_id in set(session_ids)
                }
            }
        }

    def _combine_multi_step_statistics(self, stats_list: List[Dict]) -> Dict[str, Any]:
        """Combine statistics from multiple adaptation steps"""
        if not stats_list:
            return {}

        combined = {
            'session_adaptations': stats_list[-1]['session_adaptations'],  # Latest counts
            'entropy_stats': stats_list[-1]['entropy_stats'],  # Latest entropy
            'adaptation_step': stats_list[-1]['adaptation_step'],
            'feature_change_magnitude': sum(s['feature_change_magnitude'] for s in stats_list),
            'convergence_metrics': stats_list[-1]['convergence_metrics'],
            'step_history': [s['feature_change_magnitude'] for s in stats_list]
        }

        return combined

    def reset_session_statistics(self, session_ids: Optional[List[str]] = None):
        """Reset adaptation statistics for specified sessions or all sessions"""
        if session_ids is None:
            # Reset all sessions
            self.session_adapter.session_stats.clear()
            self.session_adapter.session_sample_counts.clear()
            self.session_adapter.adaptation_history.clear()
        else:
            # Reset specific sessions
            for session_id in session_ids:
                if session_id in self.session_adapter.session_stats:
                    del self.session_adapter.session_stats[session_id]
                if session_id in self.session_adapter.session_sample_counts:
                    del self.session_adapter.session_sample_counts[session_id]
                if session_id in self.session_adapter.adaptation_history:
                    del self.session_adapter.adaptation_history[session_id]

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about all tracked sessions"""
        return {
            'num_sessions': len(self.session_adapter.session_stats),
            'session_sample_counts': dict(self.session_adapter.session_sample_counts),
            'total_adaptation_steps': self.adaptation_step_count,
            'entropy_history': list(self.entropy_minimizer.entropy_history)
        }