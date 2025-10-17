"""
Evaluation metrics for MODE-SSM training and validation.
Implements WER, mode classification accuracy, and other neural decoding metrics.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    logging.warning("jiwer not available - WER calculation will be limited")

from datasets.phoneme_vocab import PhonemeVocabulary, ctc_decode_predictions


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    # Primary metrics
    wer: float = 0.0
    cer: float = 0.0
    mode_accuracy: float = 0.0

    # Loss components
    total_loss: float = 0.0
    ctc_loss: float = 0.0
    rnnt_loss: float = 0.0
    mode_loss: float = 0.0
    flow_loss: float = 0.0

    # Detailed metrics
    num_samples: int = 0
    num_correct_modes: int = 0
    num_total_modes: int = 0

    # Per-mode metrics
    silent_mode_wer: float = 0.0
    vocalized_mode_wer: float = 0.0
    silent_mode_count: int = 0
    vocalized_mode_count: int = 0

    # Additional metrics
    phoneme_accuracy: float = 0.0
    sequence_accuracy: float = 0.0  # Exact match
    mean_edit_distance: float = 0.0

    # Timing metrics
    inference_time_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Debug info
    failed_decodings: int = 0
    empty_predictions: int = 0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class WERCalculator:
    """Word Error Rate calculation with phoneme-to-text conversion"""

    def __init__(self, use_jiwer: bool = True):
        self.use_jiwer = use_jiwer and JIWER_AVAILABLE
        self.phoneme_vocab = PhonemeVocabulary()

        if not self.use_jiwer:
            logger.warning("Using fallback WER calculation - install jiwer for better accuracy")

    def calculate_wer(
        self,
        predictions: List[List[str]],
        references: List[List[str]]
    ) -> Tuple[float, List[float]]:
        """
        Calculate Word Error Rate between predictions and references.

        Args:
            predictions: List of predicted phoneme sequences
            references: List of reference phoneme sequences

        Returns:
            (overall_wer, per_sample_wers)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        if len(predictions) == 0:
            return 0.0, []

        if self.use_jiwer:
            return self._calculate_wer_jiwer(predictions, references)
        else:
            return self._calculate_wer_fallback(predictions, references)

    def _calculate_wer_jiwer(
        self,
        predictions: List[List[str]],
        references: List[List[str]]
    ) -> Tuple[float, List[float]]:
        """Calculate WER using jiwer library"""
        # Convert phonemes to text approximations
        pred_texts = [
            self.phoneme_vocab.phonemes_to_text(pred)
            for pred in predictions
        ]
        ref_texts = [
            self.phoneme_vocab.phonemes_to_text(ref)
            for ref in references
        ]

        # Filter empty strings
        valid_pairs = [
            (pred, ref) for pred, ref in zip(pred_texts, ref_texts)
            if ref.strip() != ""
        ]

        if not valid_pairs:
            return 0.0, []

        pred_texts, ref_texts = zip(*valid_pairs)

        # Calculate overall WER
        try:
            overall_wer = jiwer.wer(list(ref_texts), list(pred_texts))
        except Exception as e:
            logger.warning(f"jiwer calculation failed: {e}, using fallback")
            return self._calculate_wer_fallback(predictions, references)

        # Calculate per-sample WER
        per_sample_wers = []
        for pred, ref in zip(pred_texts, ref_texts):
            if ref.strip() == "":
                per_sample_wers.append(0.0)
            else:
                try:
                    sample_wer = jiwer.wer([ref], [pred])
                    per_sample_wers.append(sample_wer)
                except:
                    per_sample_wers.append(1.0)  # Max error

        return overall_wer, per_sample_wers

    def _calculate_wer_fallback(
        self,
        predictions: List[List[str]],
        references: List[List[str]]
    ) -> Tuple[float, List[float]]:
        """Fallback WER calculation using edit distance on phonemes"""
        total_errors = 0
        total_ref_length = 0
        per_sample_wers = []

        for pred, ref in zip(predictions, references):
            if len(ref) == 0:
                per_sample_wers.append(0.0)
                continue

            # Calculate edit distance
            errors = self._edit_distance(pred, ref)
            wer = errors / len(ref) if len(ref) > 0 else 0.0
            per_sample_wers.append(wer)

            total_errors += errors
            total_ref_length += len(ref)

        overall_wer = total_errors / total_ref_length if total_ref_length > 0 else 0.0
        return overall_wer, per_sample_wers

    def _edit_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate edit distance between two sequences"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )

        return dp[m][n]


class ModeClassificationEvaluator:
    """Evaluator for speaking mode classification"""

    def __init__(self, num_modes: int = 2):
        self.num_modes = num_modes
        self.mode_names = ['silent', 'vocalized']

    def calculate_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate mode classification accuracy.

        Args:
            predictions: [N, num_modes] prediction logits
            targets: [N] target mode indices

        Returns:
            Dictionary with accuracy metrics
        """
        # Convert predictions to class indices
        pred_classes = torch.argmax(predictions, dim=-1)

        # Overall accuracy
        correct = (pred_classes == targets).sum().item()
        total = len(targets)
        accuracy = correct / total if total > 0 else 0.0

        # Per-class accuracy
        per_class_accuracy = {}
        for mode_idx in range(self.num_modes):
            mode_mask = targets == mode_idx
            if mode_mask.sum() > 0:
                mode_correct = (pred_classes[mode_mask] == targets[mode_mask]).sum().item()
                mode_total = mode_mask.sum().item()
                mode_accuracy = mode_correct / mode_total
                mode_name = self.mode_names[mode_idx] if mode_idx < len(self.mode_names) else f"mode_{mode_idx}"
                per_class_accuracy[f"{mode_name}_accuracy"] = mode_accuracy

        return {
            'overall_accuracy': accuracy,
            'num_correct': correct,
            'num_total': total,
            **per_class_accuracy
        }


class PhonemeAccuracyEvaluator:
    """Evaluator for phoneme-level accuracy"""

    def __init__(self, blank_idx: int = 0):
        self.blank_idx = blank_idx

    def calculate_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        ignore_blanks: bool = True
    ) -> Dict[str, float]:
        """
        Calculate phoneme-level accuracy.

        Args:
            predictions: [N, T, V] prediction logits
            targets: [N, L] target phoneme indices
            target_lengths: [N] target sequence lengths
            ignore_blanks: Whether to ignore blank tokens in accuracy

        Returns:
            Dictionary with accuracy metrics
        """
        # Get predicted classes
        pred_classes = torch.argmax(predictions, dim=-1)  # [N, T]

        total_correct = 0
        total_phonemes = 0

        for i in range(len(targets)):
            pred_seq = pred_classes[i]
            target_seq = targets[i][:target_lengths[i]]

            # Apply CTC decoding to predictions
            pred_phonemes = self._ctc_decode_single(pred_seq, ignore_blanks)
            target_phonemes = target_seq.tolist()

            # Calculate accuracy for this sequence
            min_len = min(len(pred_phonemes), len(target_phonemes))
            if min_len > 0:
                correct = sum(1 for p, t in zip(pred_phonemes, target_phonemes) if p == t)
                total_correct += correct
                total_phonemes += len(target_phonemes)

        accuracy = total_correct / total_phonemes if total_phonemes > 0 else 0.0

        return {
            'phoneme_accuracy': accuracy,
            'total_correct': total_correct,
            'total_phonemes': total_phonemes
        }

    def _ctc_decode_single(
        self,
        predictions: torch.Tensor,
        ignore_blanks: bool = True
    ) -> List[int]:
        """Simple CTC decoding for single sequence"""
        decoded = []
        prev_token = None

        for token in predictions.tolist():
            if ignore_blanks and token == self.blank_idx:
                prev_token = token
                continue

            if token != prev_token:
                decoded.append(token)

            prev_token = token

        return decoded


class EvaluationManager:
    """Main evaluation manager that orchestrates all metrics"""

    def __init__(self, vocab_size: int = 40, blank_idx: int = 0, silence_idx: int = 39):
        self.vocab_size = vocab_size
        self.blank_idx = blank_idx
        self.silence_idx = silence_idx

        # Initialize evaluators
        self.wer_calculator = WERCalculator()
        self.mode_evaluator = ModeClassificationEvaluator()
        self.phoneme_evaluator = PhonemeAccuracyEvaluator(blank_idx)
        self.phoneme_vocab = PhonemeVocabulary()

    def evaluate_batch(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        training_stage: str = "joint"
    ) -> EvaluationResults:
        """
        Evaluate a batch of predictions.

        Args:
            model_outputs: Model outputs dictionary
            batch: Input batch dictionary
            training_stage: Current training stage

        Returns:
            EvaluationResults with computed metrics
        """
        results = EvaluationResults()
        results.num_samples = len(batch['neural_features'])

        # Extract predictions and targets
        try:
            # Phoneme predictions (CTC or RNN-T)
            phoneme_predictions = None
            if 'ctc_logits' in model_outputs:
                phoneme_predictions = model_outputs['ctc_logits']
            elif 'rnnt_logits' in model_outputs:
                # For RNN-T, we need to do beam search decoding
                # For now, use CTC-like decoding
                phoneme_predictions = model_outputs['rnnt_logits']

            # Mode predictions
            mode_predictions = model_outputs.get('mode_logits')

            # Targets
            phoneme_targets = batch.get('phoneme_labels')
            target_lengths = batch.get('label_lengths')
            mode_targets = batch.get('mode_labels')  # May not exist

            # Calculate WER if we have phoneme predictions and targets
            if (phoneme_predictions is not None and
                phoneme_targets is not None and
                target_lengths is not None):

                results.wer, results.mean_edit_distance = self._calculate_wer_metrics(
                    phoneme_predictions, phoneme_targets, target_lengths
                )

                # Calculate phoneme accuracy
                phoneme_metrics = self.phoneme_evaluator.calculate_accuracy(
                    phoneme_predictions, phoneme_targets, target_lengths
                )
                results.phoneme_accuracy = phoneme_metrics['phoneme_accuracy']

            # Calculate mode classification accuracy
            if mode_predictions is not None and mode_targets is not None:
                mode_metrics = self.mode_evaluator.calculate_accuracy(
                    mode_predictions, mode_targets
                )
                results.mode_accuracy = mode_metrics['overall_accuracy']
                results.num_correct_modes = mode_metrics['num_correct']
                results.num_total_modes = mode_metrics['num_total']

            # Extract loss components
            if 'loss' in model_outputs:
                results.total_loss = model_outputs['loss'].item()
            if 'ctc_loss' in model_outputs:
                results.ctc_loss = model_outputs['ctc_loss'].item()
            if 'rnnt_loss' in model_outputs:
                results.rnnt_loss = model_outputs['rnnt_loss'].item()
            if 'mode_loss' in model_outputs:
                results.mode_loss = model_outputs['mode_loss'].item()
            if 'flow_loss' in model_outputs:
                results.flow_loss = model_outputs['flow_loss'].item()

        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}")
            results.failed_decodings = results.num_samples

        return results

    def _calculate_wer_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> Tuple[float, float]:
        """Calculate WER and edit distance metrics"""
        try:
            # Decode predictions
            sequence_lengths = torch.full(
                (predictions.shape[0],),
                predictions.shape[1],
                dtype=torch.long
            )

            predicted_phonemes = ctc_decode_predictions(
                predictions, sequence_lengths, self.blank_idx
            )

            # Extract target phonemes
            target_phonemes = []
            for i in range(len(targets)):
                seq_len = target_lengths[i].item()
                target_seq = targets[i][:seq_len]
                # Convert indices to phoneme strings
                phonemes = [
                    self.phoneme_vocab.logit_to_phoneme[idx.item()]
                    for idx in target_seq
                    if 0 <= idx.item() < len(self.phoneme_vocab.logit_to_phoneme)
                ]
                target_phonemes.append(phonemes)

            # Calculate WER
            if len(predicted_phonemes) > 0 and len(target_phonemes) > 0:
                wer, per_sample_wers = self.wer_calculator.calculate_wer(
                    predicted_phonemes, target_phonemes
                )
                mean_edit_distance = np.mean(per_sample_wers) if per_sample_wers else 0.0
                return wer, mean_edit_distance
            else:
                return 0.0, 0.0

        except Exception as e:
            logger.error(f"Error calculating WER metrics: {e}")
            return 0.0, 0.0

    def aggregate_results(self, results_list: List[EvaluationResults]) -> EvaluationResults:
        """Aggregate results across multiple batches"""
        if not results_list:
            return EvaluationResults()

        aggregated = EvaluationResults()

        # Aggregate counts
        total_samples = sum(r.num_samples for r in results_list)
        total_correct_modes = sum(r.num_correct_modes for r in results_list)
        total_modes = sum(r.num_total_modes for r in results_list)

        # Weighted averages for losses
        total_loss = sum(r.total_loss * r.num_samples for r in results_list)
        ctc_loss = sum(r.ctc_loss * r.num_samples for r in results_list)
        rnnt_loss = sum(r.rnnt_loss * r.num_samples for r in results_list)
        mode_loss = sum(r.mode_loss * r.num_samples for r in results_list)
        flow_loss = sum(r.flow_loss * r.num_samples for r in results_list)

        # WER calculation (weighted by number of samples)
        wer_sum = sum(r.wer * r.num_samples for r in results_list if r.wer > 0)
        wer_samples = sum(r.num_samples for r in results_list if r.wer > 0)

        # Phoneme accuracy (weighted average)
        phoneme_acc_sum = sum(r.phoneme_accuracy * r.num_samples for r in results_list)

        # Set aggregated values
        aggregated.num_samples = total_samples
        aggregated.wer = wer_sum / wer_samples if wer_samples > 0 else 0.0
        aggregated.mode_accuracy = total_correct_modes / total_modes if total_modes > 0 else 0.0
        aggregated.phoneme_accuracy = phoneme_acc_sum / total_samples if total_samples > 0 else 0.0

        aggregated.total_loss = total_loss / total_samples if total_samples > 0 else 0.0
        aggregated.ctc_loss = ctc_loss / total_samples if total_samples > 0 else 0.0
        aggregated.rnnt_loss = rnnt_loss / total_samples if total_samples > 0 else 0.0
        aggregated.mode_loss = mode_loss / total_samples if total_samples > 0 else 0.0
        aggregated.flow_loss = flow_loss / total_samples if total_samples > 0 else 0.0

        # Count failed decodings
        aggregated.failed_decodings = sum(r.failed_decodings for r in results_list)
        aggregated.empty_predictions = sum(r.empty_predictions for r in results_list)

        return aggregated

    def log_results(self, results: EvaluationResults, prefix: str = ""):
        """Log evaluation results"""
        logger.info(f"{prefix}Evaluation Results:")
        logger.info(f"  WER: {results.wer:.3f}")
        logger.info(f"  Mode Accuracy: {results.mode_accuracy:.3f}")
        logger.info(f"  Phoneme Accuracy: {results.phoneme_accuracy:.3f}")
        logger.info(f"  Total Loss: {results.total_loss:.4f}")

        if results.ctc_loss > 0:
            logger.info(f"  CTC Loss: {results.ctc_loss:.4f}")
        if results.rnnt_loss > 0:
            logger.info(f"  RNN-T Loss: {results.rnnt_loss:.4f}")
        if results.mode_loss > 0:
            logger.info(f"  Mode Loss: {results.mode_loss:.4f}")
        if results.flow_loss > 0:
            logger.info(f"  Flow Loss: {results.flow_loss:.4f}")

        if results.failed_decodings > 0:
            logger.warning(f"  Failed Decodings: {results.failed_decodings}")
        if results.empty_predictions > 0:
            logger.warning(f"  Empty Predictions: {results.empty_predictions}")


def create_evaluation_manager(config: Dict[str, Any]) -> EvaluationManager:
    """Create evaluation manager from configuration"""
    return EvaluationManager(
        vocab_size=config.get('vocab_size', 40),
        blank_idx=config.get('phoneme_blank_idx', 0),
        silence_idx=config.get('phoneme_silence_idx', 39)
    )