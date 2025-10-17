"""
RNNT and CTC decoder heads for MODE-SSM.
Implements joint network for RNNT and CTC projection for alignment supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union


class RNNTDecoder(nn.Module):
    """
    RNN-Transducer decoder with joint network.

    Components:
    - Predictor network (LSTM-based)
    - Joint network for encoder-predictor fusion
    - Beam search decoding support
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        predictor_layers: int = 2,
        predictor_hidden_size: int = 512,
        joint_hidden_size: int = 512,
        dropout: float = 0.1,
        beam_size: int = 4,
        blank_idx: int = 0
    ):
        """
        Initialize RNNT decoder.

        Args:
            vocab_size: Size of vocabulary (including blank)
            d_model: Encoder dimension
            predictor_layers: Number of LSTM layers in predictor
            predictor_hidden_size: Hidden size for predictor LSTM
            joint_hidden_size: Hidden size for joint network
            dropout: Dropout probability
            beam_size: Beam size for inference
            blank_idx: Index of blank token
        """
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.predictor_layers = predictor_layers
        self.predictor_hidden_size = predictor_hidden_size
        self.joint_hidden_size = joint_hidden_size
        self.beam_size = beam_size
        self.blank_idx = blank_idx

        # Embedding layer for predictor input
        self.embedding = nn.Embedding(vocab_size, predictor_hidden_size)

        # Predictor network (LSTM-based)
        self.predictor = nn.LSTM(
            input_size=predictor_hidden_size,
            hidden_size=predictor_hidden_size,
            num_layers=predictor_layers,
            dropout=dropout if predictor_layers > 1 else 0.0,
            batch_first=True
        )

        # Predictor output projection
        self.predictor_proj = nn.Linear(predictor_hidden_size, joint_hidden_size)

        # Encoder projection for joint network
        self.encoder_proj = nn.Linear(d_model, joint_hidden_size)

        # Joint network
        self.joint_network = nn.Sequential(
            nn.Linear(joint_hidden_size, joint_hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(joint_hidden_size, joint_hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(joint_hidden_size, vocab_size)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Initialize LSTM
        for name, param in self.predictor.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Initialize linear layers
        for module in [self.predictor_proj, self.encoder_proj] + list(self.joint_network.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _forward_predictor(
        self,
        targets: torch.Tensor,
        predictor_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through predictor network.

        Args:
            targets: Target token indices [batch_size, target_len]
            predictor_state: Optional initial LSTM state

        Returns:
            Tuple of (predictor_outputs, final_state)
            - predictor_outputs: [batch_size, target_len, predictor_hidden_size]
            - final_state: Final LSTM state
        """
        # Embed target tokens
        embedded = self.embedding(targets)  # [B, U, predictor_hidden_size]

        # Pass through LSTM
        lstm_output, final_state = self.predictor(embedded, predictor_state)

        return lstm_output, final_state

    def _forward_joint(
        self,
        encoder_outputs: torch.Tensor,
        predictor_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through joint network.

        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, d_model]
            predictor_outputs: Predictor outputs [batch_size, target_len, predictor_hidden_size]

        Returns:
            Joint network outputs [batch_size, seq_len, target_len, vocab_size]
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        _, target_len, _ = predictor_outputs.shape

        # Project encoder and predictor outputs
        enc_proj = self.encoder_proj(encoder_outputs)  # [B, T, joint_hidden_size]
        pred_proj = self.predictor_proj(predictor_outputs)  # [B, U, joint_hidden_size]

        # Expand dimensions for broadcasting
        enc_expanded = enc_proj.unsqueeze(2)  # [B, T, 1, joint_hidden_size]
        pred_expanded = pred_proj.unsqueeze(1)  # [B, 1, U, joint_hidden_size]

        # Combine encoder and predictor representations
        joint_input = enc_expanded + pred_expanded  # [B, T, U, joint_hidden_size]

        # Reshape for joint network
        joint_input_flat = joint_input.view(-1, self.joint_hidden_size)

        # Pass through joint network
        joint_output_flat = self.joint_network(joint_input_flat)  # [B*T*U, vocab_size]

        # Reshape back to original dimensions
        joint_output = joint_output_flat.view(batch_size, seq_len, target_len, self.vocab_size)

        return joint_output

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        targets: torch.Tensor,
        encoder_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, d_model]
            targets: Target sequences [batch_size, target_len]
            encoder_lengths: Encoder sequence lengths [batch_size]
            target_lengths: Target sequence lengths [batch_size]

        Returns:
            Dictionary containing:
            - rnnt_logits: Joint network outputs [batch_size, seq_len, target_len+1, vocab_size]
        """
        batch_size, target_len = targets.shape
        device = targets.device

        # Prepend blank token to targets for predictor
        # RNN-T predictor needs to see [blank, y1, y2, ..., yU]
        # This gives us U+1 timesteps for the predictor
        blank_prefix = torch.full((batch_size, 1), self.blank_idx, dtype=targets.dtype, device=device)
        targets_with_blank = torch.cat([blank_prefix, targets], dim=1)  # [B, U+1]

        # Forward through predictor
        predictor_outputs, _ = self._forward_predictor(targets_with_blank)

        # Forward through joint network
        joint_outputs = self._forward_joint(encoder_outputs, predictor_outputs)

        return {'rnnt_logits': joint_outputs}

    def inference(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: Optional[torch.Tensor] = None,
        max_target_len: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inference with greedy decoding.

        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, d_model]
            encoder_lengths: Encoder sequence lengths [batch_size]
            max_target_len: Maximum target sequence length

        Returns:
            Dictionary containing:
            - predictions: Predicted sequences [batch_size, max_pred_len]
            - prediction_lengths: Lengths of predictions [batch_size]
        """
        batch_size, seq_len, d_model = encoder_outputs.shape

        if max_target_len is None:
            max_target_len = seq_len  # Default max length

        # Initialize predictions
        predictions = []
        prediction_lengths = []

        # Process each sample in batch
        for b in range(batch_size):
            # Get encoder output for this sample
            enc_out = encoder_outputs[b:b+1]  # [1, T, d_model]
            enc_len = encoder_lengths[b].item() if encoder_lengths is not None else seq_len

            # Greedy decoding
            pred_seq = self._greedy_decode(enc_out, enc_len, max_target_len)

            predictions.append(pred_seq)
            prediction_lengths.append(len(pred_seq))

        # Pad sequences to same length
        max_pred_len = max(prediction_lengths) if prediction_lengths else 1
        padded_predictions = torch.zeros(batch_size, max_pred_len, dtype=torch.long, device=encoder_outputs.device)

        for b, pred_seq in enumerate(predictions):
            padded_predictions[b, :len(pred_seq)] = torch.tensor(pred_seq, device=encoder_outputs.device)

        return {
            'predictions': padded_predictions,
            'prediction_lengths': torch.tensor(prediction_lengths, device=encoder_outputs.device)
        }

    def _greedy_decode(
        self,
        encoder_output: torch.Tensor,
        encoder_length: int,
        max_target_len: int
    ) -> List[int]:
        """
        Greedy decoding for single sequence.

        Args:
            encoder_output: Encoder output [1, seq_len, d_model]
            encoder_length: Actual encoder sequence length
            max_target_len: Maximum target length

        Returns:
            Predicted sequence as list of token indices
        """
        device = encoder_output.device
        prediction = []

        # Initialize predictor state
        predictor_state = None

        # Current target context (start with blank or SOS token)
        current_target = torch.tensor([[self.blank_idx]], device=device)

        for t in range(encoder_length):
            for u in range(max_target_len):
                # Get predictor output for current target context
                pred_out, predictor_state = self._forward_predictor(current_target, predictor_state)

                # Get encoder output at current time step
                enc_t = encoder_output[:, t:t+1, :]  # [1, 1, d_model]

                # Compute joint network output
                joint_out = self._forward_joint(enc_t, pred_out)  # [1, 1, 1, vocab_size]
                logits = joint_out.squeeze()  # [vocab_size]

                # Get predicted token
                pred_token = torch.argmax(logits).item()

                if pred_token == self.blank_idx:
                    # Blank token - advance time
                    break
                else:
                    # Non-blank token - add to prediction and continue
                    prediction.append(pred_token)
                    current_target = torch.tensor([[pred_token]], device=device)

        return prediction

    def beam_search(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: Optional[torch.Tensor] = None,
        beam_size: Optional[int] = None,
        max_target_len: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Beam search decoding for RNNT.

        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, d_model]
            encoder_lengths: Encoder sequence lengths [batch_size]
            beam_size: Beam size for search
            max_target_len: Maximum target sequence length

        Returns:
            Dictionary with beam search results
        """
        if beam_size is None:
            beam_size = self.beam_size

        batch_size, seq_len, d_model = encoder_outputs.shape
        device = encoder_outputs.device

        if max_target_len is None:
            max_target_len = seq_len

        all_predictions = []
        all_prediction_lengths = []

        # Process each sample in batch separately
        for b in range(batch_size):
            enc_out = encoder_outputs[b:b+1]  # [1, T, d_model]
            enc_len = encoder_lengths[b].item() if encoder_lengths is not None else seq_len

            # Run beam search for this sample
            best_sequence = self._beam_search_single(
                enc_out, enc_len, beam_size, max_target_len
            )

            all_predictions.append(best_sequence)
            all_prediction_lengths.append(len(best_sequence))

        # Pad sequences to same length
        if all_prediction_lengths:
            max_pred_len = max(all_prediction_lengths)
        else:
            max_pred_len = 1

        padded_predictions = torch.zeros(
            batch_size, max_pred_len, dtype=torch.long, device=device
        )

        for b, pred_seq in enumerate(all_predictions):
            if len(pred_seq) > 0:
                padded_predictions[b, :len(pred_seq)] = torch.tensor(
                    pred_seq, device=device
                )

        return {
            'predictions': padded_predictions,
            'prediction_lengths': torch.tensor(all_prediction_lengths, device=device)
        }

    def _beam_search_single(
        self,
        encoder_output: torch.Tensor,
        encoder_length: int,
        beam_size: int,
        max_target_len: int
    ) -> List[int]:
        """
        Beam search for single sequence.

        Args:
            encoder_output: Encoder output [1, seq_len, d_model]
            encoder_length: Actual encoder sequence length
            beam_size: Beam size
            max_target_len: Maximum target length

        Returns:
            Best predicted sequence
        """
        device = encoder_output.device

        # Beam search state
        # Each beam contains: (sequence, log_prob, predictor_state)
        beams = [([], 0.0, None)]  # Start with empty sequence

        for t in range(encoder_length):
            new_beams = []

            # Get encoder output at current time step
            enc_t = encoder_output[:, t:t+1, :]  # [1, 1, d_model]

            for sequence, log_prob, predictor_state in beams:
                # Try advancing time (blank token)
                blank_beam = self._score_blank_token(
                    enc_t, sequence, log_prob, predictor_state
                )
                if blank_beam:
                    new_beams.append(blank_beam)

                # Try emitting tokens
                for u in range(min(len(sequence) + 1, max_target_len)):
                    token_beams = self._score_emission_tokens(
                        enc_t, sequence, log_prob, predictor_state, beam_size
                    )
                    new_beams.extend(token_beams)

                    # If we emitted a token, we can continue in the same time step
                    if token_beams:
                        # Continue with the best token beam for this timestep
                        best_token_beam = max(token_beams, key=lambda x: x[1])
                        sequence, log_prob, predictor_state = best_token_beam
                    else:
                        break

            # Keep top-k beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        # Return best sequence
        if beams:
            best_sequence, _, _ = beams[0]
            return best_sequence
        else:
            return []

    def _score_blank_token(
        self,
        enc_t: torch.Tensor,
        sequence: List[int],
        log_prob: float,
        predictor_state: Optional[Tuple]
    ) -> Optional[Tuple]:
        """Score blank token (advance time)"""
        try:
            # Get predictor output for current sequence
            if not sequence:
                # Empty sequence - use blank token as input
                current_target = torch.tensor([[self.blank_idx]], device=enc_t.device)
            else:
                # Use last token as input
                current_target = torch.tensor([[sequence[-1]]], device=enc_t.device)

            # Forward through predictor
            pred_out, new_predictor_state = self._forward_predictor(
                current_target, predictor_state
            )

            # Forward through joint network
            joint_out = self._forward_joint(enc_t, pred_out)  # [1, 1, 1, vocab_size]
            logits = joint_out.squeeze()  # [vocab_size]

            # Get log probability of blank token
            log_probs = F.log_softmax(logits, dim=-1)
            blank_log_prob = log_probs[self.blank_idx].item()

            new_log_prob = log_prob + blank_log_prob

            return (sequence.copy(), new_log_prob, new_predictor_state)

        except Exception:
            return None

    def _score_emission_tokens(
        self,
        enc_t: torch.Tensor,
        sequence: List[int],
        log_prob: float,
        predictor_state: Optional[Tuple],
        top_k: int = 5
    ) -> List[Tuple]:
        """Score emission tokens (emit new tokens)"""
        try:
            # Get predictor output
            if not sequence:
                current_target = torch.tensor([[self.blank_idx]], device=enc_t.device)
            else:
                current_target = torch.tensor([[sequence[-1]]], device=enc_t.device)

            pred_out, new_predictor_state = self._forward_predictor(
                current_target, predictor_state
            )

            # Forward through joint network
            joint_out = self._forward_joint(enc_t, pred_out)
            logits = joint_out.squeeze()  # [vocab_size]

            # Get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Get top-k non-blank tokens
            top_k_values, top_k_indices = torch.topk(log_probs, top_k)

            token_beams = []
            for i in range(top_k):
                token_idx = top_k_indices[i].item()
                token_log_prob = top_k_values[i].item()

                # Skip blank token for emission
                if token_idx == self.blank_idx:
                    continue

                new_sequence = sequence + [token_idx]
                new_log_prob = log_prob + token_log_prob

                token_beams.append((new_sequence, new_log_prob, new_predictor_state))

            return token_beams

        except Exception:
            return []


class CTCDecoder(nn.Module):
    """
    CTC decoder for alignment supervision.

    Simple linear projection from encoder outputs to vocabulary logits.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout: float = 0.1,
        blank_idx: int = 0
    ):
        """
        Initialize CTC decoder.

        Args:
            vocab_size: Size of vocabulary (including blank)
            d_model: Encoder dimension
            dropout: Dropout probability
            blank_idx: Index of blank token
        """
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.blank_idx = blank_idx

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Linear classifier
        self.classifier = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CTC decoder.

        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, d_model]
            encoder_lengths: Encoder sequence lengths [batch_size]

        Returns:
            Dictionary containing:
            - ctc_logits: CTC logits [batch_size, seq_len, vocab_size]
        """
        # Apply dropout
        features = self.dropout(encoder_outputs)

        # Linear projection to vocabulary
        ctc_logits = self.classifier(features)  # [B, T, vocab_size]

        return {'ctc_logits': ctc_logits}

    def get_log_probabilities(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get log probabilities for CTC loss computation.

        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, d_model]
            encoder_lengths: Encoder sequence lengths [batch_size]

        Returns:
            Log probabilities [batch_size, seq_len, vocab_size]
        """
        outputs = self.forward(encoder_outputs, encoder_lengths)
        return F.log_softmax(outputs['ctc_logits'], dim=-1)

    def greedy_decode(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: Optional[torch.Tensor] = None,
        remove_blanks: bool = True,
        remove_repeats: bool = True
    ) -> Dict[str, Union[torch.Tensor, List[List[int]]]]:
        """
        Greedy CTC decoding.

        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, d_model]
            encoder_lengths: Encoder sequence lengths [batch_size]
            remove_blanks: Whether to remove blank tokens
            remove_repeats: Whether to remove repeated tokens

        Returns:
            Dictionary containing:
            - predictions: Raw predictions [batch_size, seq_len]
            - decoded_sequences: Decoded sequences as list of lists
        """
        outputs = self.forward(encoder_outputs, encoder_lengths)
        ctc_logits = outputs['ctc_logits']

        # Get predictions (argmax)
        predictions = torch.argmax(ctc_logits, dim=-1)  # [B, T]

        # Decode sequences
        batch_size, seq_len = predictions.shape
        decoded_sequences = []

        for b in range(batch_size):
            pred_seq = predictions[b]
            seq_len_b = encoder_lengths[b].item() if encoder_lengths is not None else seq_len

            # Truncate to actual length
            pred_seq = pred_seq[:seq_len_b].tolist()

            # Apply CTC decoding rules
            decoded_seq = []
            prev_token = None

            for token in pred_seq:
                # Remove blanks if requested
                if remove_blanks and token == self.blank_idx:
                    prev_token = token
                    continue

                # Remove repeats if requested
                if remove_repeats and token == prev_token:
                    continue

                decoded_seq.append(token)
                prev_token = token

            decoded_sequences.append(decoded_seq)

        return {
            'predictions': predictions,
            'decoded_sequences': decoded_sequences
        }

    def extra_repr(self) -> str:
        """Extra representation string"""
        return f'vocab_size={self.vocab_size}, d_model={self.d_model}'