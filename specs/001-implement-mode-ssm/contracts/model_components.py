"""
Model Component Contracts for MODE-SSM Architecture
Defines interfaces for individual neural network components
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from enum import Enum


class SpeakingMode(Enum):
    """Speaking mode classification"""
    SILENT = 0
    VOCALIZED = 1


@dataclass
class NeuralFeatures:
    """Neural signal features with metadata"""
    features: torch.Tensor          # [B, T, 512] - Neural features
    sequence_lengths: torch.Tensor  # [B] - Valid lengths per sequence
    quality_mask: torch.Tensor      # [B, 512] - Channel quality mask
    session_metadata: Dict[str, Union[str, int]]  # Session info for TTA


@dataclass
class EncoderOutputs:
    """Encoder module outputs"""
    hidden_states: torch.Tensor     # [B, T, D] - Encoded representations
    mode_logits: torch.Tensor       # [B, 2] - Speaking mode classification
    mode_embeddings: torch.Tensor   # [B, D] - Mode conditioning embeddings
    attention_mask: torch.Tensor    # [B, T] - Sequence attention mask


@dataclass
class DecoderOutputs:
    """Decoder module outputs"""
    rnnt_logits: torch.Tensor       # [B, T, U, V] - RNNT joint network output
    ctc_logits: torch.Tensor        # [B, T, V] - CTC output logits
    prediction_network_state: Optional[torch.Tensor] = None  # RNNT hidden states


class PreprocessorInterface(ABC):
    """Interface for neural signal preprocessing"""

    @abstractmethod
    def normalize_features(self, features: torch.Tensor,
                          update_stats: bool = True) -> torch.Tensor:
        """Apply z-score normalization with running statistics"""
        pass

    @abstractmethod
    def apply_channel_gating(self, features: torch.Tensor,
                           quality_mask: torch.Tensor) -> torch.Tensor:
        """Apply learnable channel attention weights"""
        pass

    @abstractmethod
    def temporal_augmentation(self, features: torch.Tensor,
                            training: bool = True) -> torch.Tensor:
        """Apply temporal masking and time warping"""
        pass

    @abstractmethod
    def validate_sequence_length(self, features: torch.Tensor) -> torch.Tensor:
        """Enforce min/max sequence length limits (50ms-30s)"""
        pass

    @abstractmethod
    def get_feature_statistics(self) -> Dict[str, torch.Tensor]:
        """Get running mean/std statistics for reproducibility"""
        pass


class SSMEncoderInterface(ABC):
    """Interface for Mamba-based state space encoder"""

    @abstractmethod
    def forward(self, features: torch.Tensor, mode_conditioning: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through SSM layers with mode conditioning"""
        pass

    @abstractmethod
    def apply_mode_gating(self, hidden_states: torch.Tensor,
                         mode_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply mode-dependent gating to SSM outputs"""
        pass

    @abstractmethod
    def get_bidirectional_output(self, features: torch.Tensor) -> torch.Tensor:
        """Process features in both forward and backward directions"""
        pass

    @abstractmethod
    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Enable/disable gradient checkpointing for memory efficiency"""
        pass


class ModeHeadInterface(ABC):
    """Interface for speaking mode classification"""

    @abstractmethod
    def classify_mode(self, encoder_outputs: torch.Tensor,
                     sequence_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify speaking mode and return logits, probabilities"""
        pass

    @abstractmethod
    def get_mode_embeddings(self, mode_probs: torch.Tensor) -> torch.Tensor:
        """Generate mode conditioning embeddings from probabilities"""
        pass

    @abstractmethod
    def apply_contrastive_learning(self, embeddings: torch.Tensor,
                                  mode_labels: torch.Tensor) -> torch.Tensor:
        """Apply contrastive loss for cross-modal learning"""
        pass

    @abstractmethod
    def get_confidence_scores(self, mode_logits: torch.Tensor) -> torch.Tensor:
        """Calculate confidence scores for mode predictions"""
        pass


class RNNTDecoderInterface(ABC):
    """Interface for RNN-T decoder (primary)"""

    @abstractmethod
    def joint_network(self, encoder_outputs: torch.Tensor,
                     predictor_outputs: torch.Tensor) -> torch.Tensor:
        """Combine encoder and predictor outputs"""
        pass

    @abstractmethod
    def prediction_network(self, previous_labels: torch.Tensor,
                          hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions from previous labels"""
        pass

    @abstractmethod
    def beam_search_decode(self, encoder_outputs: torch.Tensor,
                          beam_size: int = 4) -> List[List[int]]:
        """Beam search decoding for inference"""
        pass

    @abstractmethod
    def greedy_decode(self, encoder_outputs: torch.Tensor) -> List[int]:
        """Greedy decoding for fast inference"""
        pass


class CTCDecoderInterface(ABC):
    """Interface for CTC decoder (auxiliary)"""

    @abstractmethod
    def forward(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """CTC projection from encoder outputs"""
        pass

    @abstractmethod
    def ctc_decode(self, logits: torch.Tensor,
                  sequence_lengths: torch.Tensor) -> List[List[int]]:
        """CTC beam search decoding"""
        pass

    @abstractmethod
    def get_alignment_path(self, logits: torch.Tensor,
                          targets: torch.Tensor) -> torch.Tensor:
        """Get CTC alignment path for analysis"""
        pass


class FlowBridgeInterface(ABC):
    """Interface for optional diffusion/flow denoising bridge"""

    @abstractmethod
    def denoise_representations(self, encoder_outputs: torch.Tensor,
                              noise_level: float = 0.1) -> torch.Tensor:
        """Apply denoising to encoder representations"""
        pass

    @abstractmethod
    def sample_noise_schedule(self, batch_size: int) -> torch.Tensor:
        """Sample noise schedule for training"""
        pass

    @abstractmethod
    def compute_denoising_loss(self, clean: torch.Tensor,
                             noisy: torch.Tensor) -> torch.Tensor:
        """Compute denoising objective loss"""
        pass

    @abstractmethod
    def enable_flow_bridge(self, enabled: bool = True) -> None:
        """Enable/disable flow bridge during training"""
        pass


class LMFusionInterface(ABC):
    """Interface for language model fusion"""

    @abstractmethod
    def load_corpus_lm(self, corpus_type: str) -> None:
        """Load corpus-specific language model"""
        pass

    @abstractmethod
    def apply_lm_rescoring(self, decoder_logits: torch.Tensor,
                          lm_logits: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
        """Apply LM rescoring to decoder outputs"""
        pass

    @abstractmethod
    def get_lm_logits(self, token_sequence: List[int]) -> torch.Tensor:
        """Get language model logits for token sequence"""
        pass

    @abstractmethod
    def mode_conditioned_fusion(self, decoder_logits: torch.Tensor,
                               lm_logits: torch.Tensor,
                               mode_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply mode-conditioned LM fusion"""
        pass


class TTAModuleInterface(ABC):
    """Interface for test-time adaptation"""

    @abstractmethod
    def adapt_to_session(self, session_data: List[torch.Tensor]) -> None:
        """Adapt model parameters to new recording session"""
        pass

    @abstractmethod
    def update_feature_statistics(self, features: torch.Tensor) -> None:
        """Update EMA feature statistics for new session"""
        pass

    @abstractmethod
    def entropy_minimization_step(self, model_outputs: torch.Tensor,
                                 learning_rate: float = 1e-5) -> torch.Tensor:
        """Single entropy minimization adaptation step"""
        pass

    @abstractmethod
    def reset_adaptation_state(self) -> None:
        """Reset adaptation state for new session"""
        pass

    @abstractmethod
    def get_adaptation_metrics(self) -> Dict[str, float]:
        """Get metrics for adaptation effectiveness"""
        pass


# Model Component Factory Functions

def create_preprocessor(d_model: int = 512, num_channels: int = 512) -> PreprocessorInterface:
    """Create preprocessor component"""
    pass


def create_ssm_encoder(d_model: int = 512, d_state: int = 64,
                      n_layers: int = 8) -> SSMEncoderInterface:
    """Create Mamba-based SSM encoder"""
    pass


def create_mode_head(d_model: int = 512, num_modes: int = 2) -> ModeHeadInterface:
    """Create speaking mode classification head"""
    pass


def create_rnnt_decoder(d_model: int = 512, vocab_size: int = 40) -> RNNTDecoderInterface:
    """Create RNN-T decoder component"""
    pass


def create_ctc_decoder(d_model: int = 512, vocab_size: int = 40) -> CTCDecoderInterface:
    """Create CTC decoder component"""
    pass


def create_flow_bridge(d_model: int = 512) -> FlowBridgeInterface:
    """Create optional flow-based denoising bridge"""
    pass


def create_lm_fusion(vocab_size: int = 40) -> LMFusionInterface:
    """Create language model fusion component"""
    pass


def create_tta_module() -> TTAModuleInterface:
    """Create test-time adaptation module"""
    pass


# Component Integration Contracts

class MODESSMModel(nn.Module):
    """Main MODE-SSM model integrating all components"""

    def __init__(self,
                 preprocessor: PreprocessorInterface,
                 encoder: SSMEncoderInterface,
                 mode_head: ModeHeadInterface,
                 rnnt_decoder: RNNTDecoderInterface,
                 ctc_decoder: CTCDecoderInterface,
                 flow_bridge: Optional[FlowBridgeInterface] = None,
                 lm_fusion: Optional[LMFusionInterface] = None,
                 tta_module: Optional[TTAModuleInterface] = None):
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.mode_head = mode_head
        self.rnnt_decoder = rnnt_decoder
        self.ctc_decoder = ctc_decoder
        self.flow_bridge = flow_bridge
        self.lm_fusion = lm_fusion
        self.tta_module = tta_module

    def forward(self, neural_features: NeuralFeatures,
               training_stage: str = "joint_train") -> DecoderOutputs:
        """Forward pass through complete model"""
        # Preprocessing
        processed_features = self.preprocessor.normalize_features(
            neural_features.features)
        processed_features = self.preprocessor.apply_channel_gating(
            processed_features, neural_features.quality_mask)

        if self.training:
            processed_features = self.preprocessor.temporal_augmentation(
                processed_features)

        # Mode classification
        mode_logits, mode_probs = self.mode_head.classify_mode(
            processed_features, neural_features.sequence_lengths)
        mode_embeddings = self.mode_head.get_mode_embeddings(mode_probs)

        # Encoding
        encoder_outputs = self.encoder.forward(
            processed_features, mode_embeddings,
            neural_features.sequence_lengths)

        # Optional denoising
        if self.flow_bridge is not None and training_stage == "denoise_train":
            encoder_outputs = self.flow_bridge.denoise_representations(
                encoder_outputs)

        # Decoding
        rnnt_logits = self.rnnt_decoder.joint_network(encoder_outputs, None)
        ctc_logits = self.ctc_decoder.forward(encoder_outputs)

        # Optional LM fusion
        if self.lm_fusion is not None:
            rnnt_logits = self.lm_fusion.mode_conditioned_fusion(
                rnnt_logits, None, mode_embeddings)

        return DecoderOutputs(
            rnnt_logits=rnnt_logits,
            ctc_logits=ctc_logits
        )

    def apply_test_time_adaptation(self, session_data: NeuralFeatures) -> None:
        """Apply test-time adaptation for neural drift"""
        if self.tta_module is not None:
            self.tta_module.adapt_to_session([session_data.features])

    def get_text_predictions(self, neural_features: NeuralFeatures) -> List[str]:
        """Generate text predictions from neural features"""
        outputs = self.forward(neural_features)
        # Decode logits to text using beam search or greedy decoding
        # Implementation depends on specific decoding strategy
        pass