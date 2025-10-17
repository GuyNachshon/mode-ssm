"""
Main MODE-SSM model integrating all components.
Mode-Aware State-Space Decoder for Brain-to-Text Translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from .preprocessor import NeuralPreprocessor
from .ssm_encoder import MambaEncoder
from .mode_head import ModeClassificationHead
from .rnnt_ctc_heads import RNNTDecoder, CTCDecoder
from .denoise_flow import FlowBridgeDenoiser, FlowBridgeConfig
from .lm_fusion import LanguageModelFusion, LMFusionConfig


@dataclass
class MODESSMConfig:
    """Configuration for MODE-SSM model"""
    # Model dimensions
    d_model: int = 512
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2

    # Preprocessor config
    num_channels: int = 512
    preprocessor_dropout: float = 0.1
    channel_attention: bool = True
    conv_kernel_size: int = 7

    # Encoder config
    encoder_layers: int = 8
    encoder_bidirectional: bool = True
    encoder_dropout: float = 0.1
    gradient_checkpointing: bool = False

    # Mode head config
    num_modes: int = 2
    mode_head_dropout: float = 0.1
    contrastive_learning: bool = True
    pooling_type: str = "global_avg"

    # Decoder configs
    vocab_size: int = 40
    predictor_layers: int = 2
    predictor_hidden_size: int = 512
    joint_hidden_size: int = 512
    decoder_dropout: float = 0.1
    beam_size: int = 4
    blank_idx: int = 0
    silence_idx: int = 39

    # Training behavior
    mixed_precision: bool = True
    compile_model: bool = False

    # Advanced features (optional)
    use_flow_bridge: bool = False
    use_lm_fusion: bool = False

    # Flow bridge config
    flow_bridge: Optional[FlowBridgeConfig] = None

    # LM fusion config
    lm_fusion: Optional[LMFusionConfig] = None


class MODESSMModel(nn.Module):
    """
    MODE-SSM: Mode-Aware State-Space Decoder for Brain-to-Text Translation.

    Architecture:
    1. NeuralPreprocessor: Signal conditioning and normalization
    2. MambaEncoder: Bidirectional state-space modeling
    3. ModeClassificationHead: Speaking mode classification
    4. RNNTDecoder: Sequence-to-sequence prediction
    5. CTCDecoder: Alignment supervision
    6. FlowBridgeDenoiser: Diffusion-based denoising (optional)
    7. LanguageModelFusion: External LM integration (optional)

    The model can operate in different training stages with selective component activation.
    Advanced features include flow bridge denoising and language model fusion.
    """

    def __init__(
        self,
        config: Optional[MODESSMConfig] = None,
        **kwargs
    ):
        """
        Initialize MODE-SSM model.

        Args:
            config: Model configuration
            **kwargs: Override config parameters
        """
        super().__init__()

        # Use default config if none provided
        if config is None:
            config = MODESSMConfig()

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # Initialize components
        self._build_model()

        # Initialize weights
        self.apply(self._init_weights)

    def _build_model(self):
        """Build all model components"""
        config = self.config

        # Neural signal preprocessor
        self.preprocessor = NeuralPreprocessor(
            num_channels=config.num_channels,
            d_model=config.d_model,
            normalization_momentum=0.1,
            channel_attention=config.channel_attention,
            conv_kernel_size=config.conv_kernel_size,
            dropout=config.preprocessor_dropout
        )

        # Bidirectional Mamba encoder
        self.encoder = MambaEncoder(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            n_layers=config.encoder_layers,
            bidirectional=config.encoder_bidirectional,
            dropout=config.encoder_dropout,
            gradient_checkpointing=config.gradient_checkpointing
        )

        # Speaking mode classification head
        self.mode_head = ModeClassificationHead(
            d_model=config.d_model,
            num_modes=config.num_modes,
            dropout=config.mode_head_dropout,
            contrastive_learning=config.contrastive_learning,
            pooling_type=config.pooling_type
        )

        # RNN-T decoder for sequence prediction
        self.rnnt_decoder = RNNTDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            predictor_layers=config.predictor_layers,
            predictor_hidden_size=config.predictor_hidden_size,
            joint_hidden_size=config.joint_hidden_size,
            dropout=config.decoder_dropout,
            beam_size=config.beam_size,
            blank_idx=config.blank_idx
        )

        # CTC decoder for alignment supervision
        self.ctc_decoder = CTCDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            dropout=config.decoder_dropout,
            blank_idx=config.blank_idx
        )

        # Optional flow bridge for denoising
        self.flow_bridge = None
        if config.use_flow_bridge and config.flow_bridge is not None:
            self.flow_bridge = FlowBridgeDenoiser(config.flow_bridge)

        # Optional language model fusion
        self.lm_fusion = None
        if config.use_lm_fusion and config.lm_fusion is not None:
            self.lm_fusion = LanguageModelFusion(config.lm_fusion)

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        neural_features: torch.Tensor,
        sequence_lengths: torch.Tensor,
        phoneme_labels: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        mode_labels: Optional[torch.Tensor] = None,
        training_stage: str = "joint",
        return_all_outputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MODE-SSM model.

        Args:
            neural_features: Neural signal features [batch_size, seq_len, num_channels]
            sequence_lengths: Actual sequence lengths [batch_size]
            phoneme_labels: Target phoneme sequences [batch_size, label_len]
            label_lengths: Target sequence lengths [batch_size]
            mode_labels: Speaking mode labels [batch_size]
            training_stage: Current training stage
            return_all_outputs: Whether to return all intermediate outputs

        Returns:
            Dictionary containing model outputs based on training stage
        """
        batch_size, seq_len, num_channels = neural_features.shape

        # Create sequence mask
        mask = self._create_sequence_mask(sequence_lengths, seq_len)

        outputs = {}

        # 1. Neural signal preprocessing
        preprocessed_features = self.preprocessor(neural_features)

        # 2. Bidirectional state-space encoding
        encoded_features = self.encoder(preprocessed_features, mask=mask)

        # Store intermediate outputs if requested
        if return_all_outputs:
            outputs.update({
                'preprocessed_features': preprocessed_features,
                'encoded_features': encoded_features
            })

        # 3. Mode classification (if enabled in training stage)
        if training_stage in ['mode', 'denoise'] or return_all_outputs:
            mode_outputs = self.mode_head(encoded_features, mask=mask)
            outputs.update(mode_outputs)

        # 4. CTC decoder (if enabled in training stage)
        if training_stage in ['ctc_warmup', 'joint', 'mode', 'denoise'] or return_all_outputs:
            ctc_outputs = self.ctc_decoder(encoded_features, sequence_lengths)
            outputs.update(ctc_outputs)

        # 5. RNN-T decoder (if enabled in training stage)
        if training_stage in ['joint', 'mode', 'denoise'] or return_all_outputs:
            if phoneme_labels is not None:
                rnnt_outputs = self.rnnt_decoder(
                    encoded_features,
                    phoneme_labels,
                    sequence_lengths,
                    label_lengths
                )
                outputs.update(rnnt_outputs)

        # 6. Flow bridge denoising (if enabled)
        if training_stage == 'denoise' and self.flow_bridge is not None:
            # Apply flow bridge denoising to preprocessed features
            flow_outputs = self.flow_bridge(preprocessed_features)
            outputs['flow_loss'] = flow_outputs['loss']
            outputs['flow_denoised'] = flow_outputs['model_pred']

            # Use denoised features for downstream processing if available
            if 'model_pred' in flow_outputs:
                # Re-encode with denoised features for improved representations
                denoised_encoded = self.encoder(flow_outputs['model_pred'], mask=mask)
                outputs['denoised_encoded_features'] = denoised_encoded

        return outputs

    def inference(
        self,
        neural_features: torch.Tensor,
        sequence_lengths: torch.Tensor,
        decode_mode: str = "greedy",
        return_mode_predictions: bool = True
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Inference mode for generating predictions.

        Args:
            neural_features: Neural signal features [batch_size, seq_len, num_channels]
            sequence_lengths: Actual sequence lengths [batch_size]
            decode_mode: Decoding mode ("greedy", "beam_search")
            return_mode_predictions: Whether to return mode predictions

        Returns:
            Dictionary with predictions and confidence scores
        """
        self.eval()

        with torch.no_grad():
            batch_size, seq_len, num_channels = neural_features.shape

            # Create sequence mask
            mask = self._create_sequence_mask(sequence_lengths, seq_len)

            # Forward pass through preprocessing and encoding
            preprocessed_features = self.preprocessor(neural_features)

            # Optional flow bridge denoising for inference
            if self.flow_bridge is not None:
                denoised_features = self.flow_bridge.denoise(preprocessed_features)
                encoded_features = self.encoder(denoised_features, mask=mask)
            else:
                encoded_features = self.encoder(preprocessed_features, mask=mask)

            results = {}

            # Mode classification
            if return_mode_predictions:
                mode_outputs = self.mode_head(encoded_features, mask=mask)
                mode_probs = F.softmax(mode_outputs['mode_logits'], dim=-1)
                mode_predictions = torch.argmax(mode_outputs['mode_logits'], dim=-1)

                results.update({
                    'mode_predictions': mode_predictions,
                    'mode_probabilities': mode_probs,
                    'mode_confidence': torch.max(mode_probs, dim=-1)[0]
                })

            # Sequence prediction
            if decode_mode == "greedy":
                # Use CTC greedy decoding for simplicity
                ctc_results = self.ctc_decoder.greedy_decode(
                    encoded_features,
                    sequence_lengths,
                    remove_blanks=True,
                    remove_repeats=True
                )
                results.update({
                    'predictions': ctc_results['predictions'],
                    'decoded_sequences': ctc_results['decoded_sequences']
                })
            elif decode_mode == "beam_search":
                # Use RNN-T beam search
                rnnt_results = self.rnnt_decoder.beam_search(
                    encoded_features,
                    sequence_lengths
                )

                # Apply LM fusion if available
                if self.lm_fusion is not None and 'beam_scores' in rnnt_results:
                    # Convert decoded sequences to text for LM fusion
                    # This is a simplified integration - full implementation would
                    # require integration within the beam search loop
                    pass

                results.update(rnnt_results)

            return results

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float],
        training_stage: str = "joint"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses based on current stage.

        Args:
            outputs: Model outputs
            targets: Ground truth targets
            loss_weights: Loss weighting for current stage
            training_stage: Current training stage

        Returns:
            Dictionary with computed losses
        """
        losses = {}
        total_loss = 0.0

        # CTC loss
        if 'ctc_logits' in outputs and loss_weights.get('ctc_weight', 0) > 0:
            ctc_loss = self._compute_ctc_loss(
                outputs['ctc_logits'],
                targets.get('phoneme_labels'),
                targets.get('sequence_lengths'),
                targets.get('label_lengths')
            )
            losses['ctc_loss'] = ctc_loss
            total_loss += loss_weights['ctc_weight'] * ctc_loss

        # RNN-T loss
        if 'rnnt_logits' in outputs and loss_weights.get('rnnt_weight', 0) > 0:
            rnnt_loss = self._compute_rnnt_loss(
                outputs['rnnt_logits'],
                targets.get('phoneme_labels'),
                targets.get('sequence_lengths'),
                targets.get('label_lengths')
            )
            losses['rnnt_loss'] = rnnt_loss
            total_loss += loss_weights['rnnt_weight'] * rnnt_loss

        # Mode classification loss
        if 'mode_logits' in outputs and loss_weights.get('mode_weight', 0) > 0:
            mode_loss = self._compute_mode_loss(
                outputs['mode_logits'],
                targets.get('mode_labels')
            )
            losses['mode_loss'] = mode_loss
            total_loss += loss_weights['mode_weight'] * mode_loss

            # Add contrastive loss if available
            if 'contrastive_features' in outputs:
                contrastive_loss = self.mode_head.compute_contrastive_loss(
                    outputs['contrastive_features'],
                    targets.get('mode_labels')
                )
                losses['contrastive_loss'] = contrastive_loss
                total_loss += loss_weights['mode_weight'] * 0.1 * contrastive_loss

        # Flow bridge loss
        if 'flow_loss' in outputs and loss_weights.get('flow_weight', 0) > 0:
            flow_loss = outputs['flow_loss']
            losses['flow_loss'] = flow_loss
            total_loss += loss_weights['flow_weight'] * flow_loss

        losses['total_loss'] = total_loss
        return losses

    def _compute_ctc_loss(
        self,
        ctc_logits: torch.Tensor,
        targets: Optional[torch.Tensor],
        sequence_lengths: Optional[torch.Tensor],
        target_lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute CTC loss"""
        if targets is None or sequence_lengths is None or target_lengths is None:
            return torch.tensor(0.0, device=ctc_logits.device)

        # Convert to log probabilities
        log_probs = F.log_softmax(ctc_logits, dim=-1)

        # Transpose for CTC loss: [T, N, C]
        log_probs = log_probs.transpose(0, 1)

        # Compute CTC loss
        ctc_loss = F.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=sequence_lengths,
            target_lengths=target_lengths,
            blank=self.config.blank_idx,
            reduction='mean',
            zero_infinity=True
        )

        return ctc_loss

    def _compute_rnnt_loss(
        self,
        rnnt_logits: torch.Tensor,
        targets: Optional[torch.Tensor],
        sequence_lengths: Optional[torch.Tensor],
        target_lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute RNN-T loss using torchaudio with adaptive scaling.

        Args:
            rnnt_logits: Joint network outputs [batch_size, seq_len, target_len, vocab_size]
            targets: Target phoneme sequences [batch_size, target_len]
            sequence_lengths: Encoder sequence lengths [batch_size]
            target_lengths: Target sequence lengths [batch_size]

        Returns:
            RNN-T loss scalar (scaled to be comparable to CTC loss)
        """
        if targets is None or sequence_lengths is None or target_lengths is None:
            return torch.tensor(0.0, device=rnnt_logits.device, requires_grad=True)

        try:
            # Import torchaudio RNN-T loss
            from torchaudio.functional import rnnt_loss

            # Convert logits to log probabilities
            # rnnt_loss expects log probabilities in shape [B, T, U, V]
            log_probs = F.log_softmax(rnnt_logits, dim=-1)

            # torchaudio rnnt_loss expects:
            # - logits: [B, T, U, V] log probabilities
            # - targets: [B, U] target sequences (WITHOUT blank tokens)
            # - logit_lengths: [B] encoder lengths
            # - target_lengths: [B] target lengths

            loss = rnnt_loss(
                logits=log_probs,
                targets=targets.int(),
                logit_lengths=sequence_lengths.int(),
                target_lengths=target_lengths.int(),
                blank=self.config.blank_idx,
                reduction='mean'
            )

            # Scale RNN-T loss to be comparable to CTC loss
            # RNN-T naturally produces losses ~100x larger due to B×T×U tensor
            # Scale by sequence length AND target length to normalize
            avg_seq_len = sequence_lengths.float().mean()
            avg_target_len = target_lengths.float().mean()

            # Very aggressive scaling for stable training across all sessions
            # Scale by sqrt(T * U) to account for the B×T×U joint tensor size
            scale_factor = torch.sqrt(avg_seq_len * avg_target_len)
            scale_factor = torch.clamp(scale_factor, min=5.0)  # Minimum scaling of 5x

            scaled_loss = loss / scale_factor

            return scaled_loss

        except Exception as e:
            # If RNN-T loss fails, log warning and return zero loss
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"RNN-T loss computation failed: {e}. Returning zero loss.")
            return torch.tensor(0.0, device=rnnt_logits.device, requires_grad=True)

    def _compute_mode_loss(
        self,
        mode_logits: torch.Tensor,
        mode_labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute mode classification loss"""
        if mode_labels is None:
            return torch.tensor(0.0, device=mode_logits.device)

        return F.cross_entropy(mode_logits, mode_labels)

    def _create_sequence_mask(
        self,
        sequence_lengths: torch.Tensor,
        max_len: int
    ) -> torch.Tensor:
        """Create boolean mask for variable-length sequences"""
        batch_size = sequence_lengths.size(0)
        mask = torch.arange(max_len, device=sequence_lengths.device).expand(
            batch_size, max_len
        ) < sequence_lengths.unsqueeze(1)
        return mask

    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size(self) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.encoder.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        self.encoder.gradient_checkpointing = False

    def freeze_component(self, component_name: str):
        """Freeze a specific component"""
        component = getattr(self, component_name, None)
        if component is not None:
            for param in component.parameters():
                param.requires_grad = False

    def unfreeze_component(self, component_name: str):
        """Unfreeze a specific component"""
        component = getattr(self, component_name, None)
        if component is not None:
            for param in component.parameters():
                param.requires_grad = True

    def extra_repr(self) -> str:
        """Extra representation string"""
        return (
            f'd_model={self.config.d_model}, '
            f'vocab_size={self.config.vocab_size}, '
            f'num_modes={self.config.num_modes}'
        )