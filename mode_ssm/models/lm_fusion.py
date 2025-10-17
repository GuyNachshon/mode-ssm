"""
Corpus-aware language model fusion for MODE-SSM.
Integrates external language models to improve decoding accuracy using contextual knowledge.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False

from datasets.phoneme_vocab import PhonemeVocabulary, LOGIT_TO_PHONEME

logger = logging.getLogger(__name__)


@dataclass
class LMFusionConfig:
    """Configuration for language model fusion"""

    # Language model settings
    lm_model_name: str = "microsoft/DialoGPT-small"  # Hugging Face model or path
    lm_weight: float = 0.3  # Weight for LM scores
    neural_weight: float = 0.7  # Weight for neural decoder scores

    # Fusion strategy
    fusion_method: str = "shallow"  # shallow, deep, attention
    fusion_layers: List[int] = None  # Layers for deep fusion
    beam_fusion: bool = True  # Apply fusion during beam search

    # Text processing
    max_context_length: int = 128
    context_window: int = 10  # Number of previous words for context
    use_corpus_adaptation: bool = True
    corpus_path: Optional[str] = None

    # Performance settings
    cache_size: int = 10000
    batch_inference: bool = True
    device_map: str = "auto"

    # Adaptation settings
    adaptation_enabled: bool = True
    adaptation_lr: float = 1e-5
    adaptation_steps: int = 100


class PhonemeToTextConverter:
    """Converts phoneme sequences to text for LM processing"""

    def __init__(self, phoneme_vocab: Optional[PhonemeVocabulary] = None):
        self.phoneme_vocab = phoneme_vocab or PhonemeVocabulary()

        # Create phoneme-to-word mapping (simplified)
        self.phoneme_to_text_map = self._create_phoneme_text_mapping()

    def _create_phoneme_text_mapping(self) -> Dict[str, str]:
        """Create mapping from phonemes to approximate text"""
        # Simplified phoneme to text mapping
        mapping = {
            'AA': 'a', 'AE': 'a', 'AH': 'a', 'AO': 'o', 'AW': 'ow', 'AY': 'i',
            'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'th', 'EH': 'e', 'ER': 'er',
            'EY': 'ay', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'i', 'IY': 'ee',
            'JH': 'j', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ng',
            'OW': 'o', 'OY': 'oy', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'sh',
            'T': 't', 'TH': 'th', 'UH': 'u', 'UW': 'oo', 'V': 'v', 'W': 'w',
            'Y': 'y', 'Z': 'z', 'ZH': 'zh', 'BLANK': '', 'SIL': ' '
        }
        return mapping

    def phonemes_to_text(self, phoneme_sequence: List[str]) -> str:
        """Convert phoneme sequence to approximate text"""
        words = []
        current_word = ""

        for phoneme in phoneme_sequence:
            if phoneme in self.phoneme_to_text_map:
                text_part = self.phoneme_to_text_map[phoneme]
                if phoneme == 'SIL':  # Silence indicates word boundary
                    if current_word:
                        words.append(current_word)
                        current_word = ""
                else:
                    current_word += text_part

        # Add final word
        if current_word:
            words.append(current_word)

        return ' '.join(words)

    def indices_to_text(self, phoneme_indices: List[int]) -> str:
        """Convert phoneme indices to text"""
        phonemes = [
            LOGIT_TO_PHONEME[idx] for idx in phoneme_indices
            if 0 <= idx < len(LOGIT_TO_PHONEME)
        ]
        return self.phonemes_to_text(phonemes)


class CorpusAdapter:
    """Adapts language model to domain-specific corpus"""

    def __init__(self, config: LMFusionConfig):
        self.config = config
        self.corpus_stats = {}
        self.word_frequencies = {}
        self.bigram_probs = {}

    def load_corpus(self, corpus_path: str):
        """Load and analyze domain corpus"""
        logger.info(f"Loading corpus from: {corpus_path}")

        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            logger.warning(f"Corpus file not found: {corpus_path}")
            return

        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                texts = f.readlines()

            self._analyze_corpus(texts)
            logger.info(f"Analyzed corpus: {len(texts)} texts, {len(self.word_frequencies)} unique words")

        except Exception as e:
            logger.error(f"Error loading corpus: {e}")

    def _analyze_corpus(self, texts: List[str]):
        """Analyze corpus for word frequencies and bigrams"""
        word_counts = {}
        bigram_counts = {}
        total_words = 0

        for text in texts:
            words = text.strip().lower().split()

            # Count words
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                total_words += 1

            # Count bigrams
            for i in range(len(words) - 1):
                bigram = (words[i], words[i + 1])
                bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        # Convert to probabilities
        self.word_frequencies = {
            word: count / total_words
            for word, count in word_counts.items()
        }

        # Compute bigram probabilities
        for (w1, w2), count in bigram_counts.items():
            if w1 not in self.bigram_probs:
                self.bigram_probs[w1] = {}
            self.bigram_probs[w1][w2] = count / word_counts[w1]

        self.corpus_stats = {
            'total_words': total_words,
            'unique_words': len(word_counts),
            'vocab_size': len(self.word_frequencies)
        }

    def get_word_score(self, word: str) -> float:
        """Get corpus-based score for word"""
        return self.word_frequencies.get(word.lower(), 1e-6)

    def get_bigram_score(self, word1: str, word2: str) -> float:
        """Get bigram probability from corpus"""
        word1 = word1.lower()
        word2 = word2.lower()

        if word1 in self.bigram_probs and word2 in self.bigram_probs[word1]:
            return self.bigram_probs[word1][word2]
        else:
            return self.word_frequencies.get(word2, 1e-6)


class LanguageModelFusion(nn.Module):
    """Language model fusion component"""

    def __init__(self, config: LMFusionConfig):
        super().__init__()
        self.config = config

        # Check transformers availability
        if not HF_TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available. LM fusion will be limited.")
            self.lm_model = None
            self.tokenizer = None
        else:
            self._load_language_model()

        # Components
        self.phoneme_converter = PhonemeToTextConverter()
        self.corpus_adapter = CorpusAdapter(config)

        # Load corpus if provided
        if config.corpus_path:
            self.corpus_adapter.load_corpus(config.corpus_path)

        # Fusion layers for deep fusion
        if config.fusion_method == "deep":
            self.fusion_layers = nn.ModuleDict()
            if config.fusion_layers:
                for layer_idx in config.fusion_layers:
                    self.fusion_layers[f"layer_{layer_idx}"] = nn.Linear(
                        config.lm_hidden_dim + config.neural_dim,
                        config.neural_dim
                    )
        elif config.fusion_method == "attention":
            self.attention_fusion = nn.MultiheadAttention(
                embed_dim=config.neural_dim,
                num_heads=8,
                batch_first=True
            )

        # Caching for efficiency
        self.lm_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _load_language_model(self):
        """Load language model from Hugging Face"""
        try:
            logger.info(f"Loading language model: {self.config.lm_model_name}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.lm_model_name,
                trust_remote_code=True
            )

            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.lm_model = AutoModelForCausalLM.from_pretrained(
                self.config.lm_model_name,
                device_map=self.config.device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            self.lm_model.eval()
            logger.info("Language model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading language model: {e}")
            self.lm_model = None
            self.tokenizer = None

    def shallow_fusion(
        self,
        neural_logits: torch.Tensor,
        context: List[str],
        candidate_texts: List[str]
    ) -> torch.Tensor:
        """
        Shallow fusion: interpolate neural and LM scores.

        Args:
            neural_logits: Neural decoder logits [batch_size, vocab_size]
            context: Previous context words
            candidate_texts: Candidate text sequences

        Returns:
            Fused logits [batch_size, vocab_size]
        """
        if self.lm_model is None:
            return neural_logits

        batch_size, vocab_size = neural_logits.shape
        device = neural_logits.device

        # Get LM scores for candidates
        lm_scores = torch.zeros_like(neural_logits)

        for batch_idx, candidate_list in enumerate(candidate_texts):
            if isinstance(candidate_list, str):
                candidate_list = [candidate_list]

            for vocab_idx, candidate in enumerate(candidate_list):
                if vocab_idx >= vocab_size:
                    break

                # Prepare context + candidate
                full_text = ' '.join(context[-self.config.context_window:] + [candidate])

                # Get LM score (cached)
                lm_score = self._get_lm_score(full_text, candidate)
                lm_scores[batch_idx, vocab_idx] = lm_score

        # Apply corpus adaptation if enabled
        if self.config.use_corpus_adaptation:
            corpus_scores = self._get_corpus_scores(context, candidate_texts)
            lm_scores = lm_scores + 0.1 * corpus_scores

        # Interpolate scores
        neural_scores = F.log_softmax(neural_logits, dim=-1)
        lm_scores = F.log_softmax(lm_scores, dim=-1)

        fused_scores = (
            self.config.neural_weight * neural_scores +
            self.config.lm_weight * lm_scores
        )

        return fused_scores

    def deep_fusion(
        self,
        neural_hidden: torch.Tensor,
        neural_logits: torch.Tensor,
        context: List[str],
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deep fusion: fuse at hidden layer level.

        Args:
            neural_hidden: Neural hidden states [batch_size, seq_len, hidden_dim]
            neural_logits: Neural logits [batch_size, vocab_size]
            context: Context words
            layer_idx: Layer index for fusion

        Returns:
            Fused hidden states and logits
        """
        if self.lm_model is None or f"layer_{layer_idx}" not in self.fusion_layers:
            return neural_hidden, neural_logits

        # Get LM hidden states
        context_text = ' '.join(context[-self.config.context_window:])
        lm_hidden = self._get_lm_hidden(context_text)

        if lm_hidden is not None:
            # Concatenate and project
            batch_size, seq_len = neural_hidden.shape[:2]
            lm_hidden_expanded = lm_hidden.unsqueeze(1).expand(-1, seq_len, -1)

            combined = torch.cat([neural_hidden, lm_hidden_expanded], dim=-1)
            fused_hidden = self.fusion_layers[f"layer_{layer_idx}"](combined)

            # Update hidden states
            neural_hidden = neural_hidden + fused_hidden

        return neural_hidden, neural_logits

    def attention_fusion(
        self,
        neural_hidden: torch.Tensor,
        neural_logits: torch.Tensor,
        context: List[str]
    ) -> torch.Tensor:
        """
        Attention-based fusion between neural and LM representations.

        Args:
            neural_hidden: Neural hidden states [batch_size, seq_len, hidden_dim]
            neural_logits: Neural logits [batch_size, vocab_size]
            context: Context words

        Returns:
            Fused logits
        """
        if self.lm_model is None:
            return neural_logits

        # Get LM hidden states
        context_text = ' '.join(context[-self.config.context_window:])
        lm_hidden = self._get_lm_hidden(context_text)

        if lm_hidden is not None:
            # Attention between neural and LM hidden states
            batch_size, seq_len, hidden_dim = neural_hidden.shape
            lm_hidden_expanded = lm_hidden.unsqueeze(1).expand(-1, seq_len, -1)

            # Apply attention
            attended_features, _ = self.attention_fusion(
                query=neural_hidden,
                key=lm_hidden_expanded,
                value=lm_hidden_expanded
            )

            # Combine with original neural features
            combined_hidden = neural_hidden + attended_features

            # Project to vocabulary
            # Note: This assumes access to output projection layer
            # In practice, would need to integrate with specific model architecture

        return neural_logits

    def beam_search_fusion(
        self,
        beam_scores: torch.Tensor,
        beam_sequences: List[List[str]],
        context: List[str]
    ) -> torch.Tensor:
        """
        Apply LM fusion during beam search.

        Args:
            beam_scores: Current beam scores [batch_size, beam_size]
            beam_sequences: Beam sequences as text
            context: Context words

        Returns:
            Updated beam scores
        """
        if self.lm_model is None:
            return beam_scores

        batch_size, beam_size = beam_scores.shape

        # Get LM scores for each beam
        for batch_idx in range(batch_size):
            for beam_idx in range(beam_size):
                sequence = beam_sequences[batch_idx][beam_idx]

                # Create full context + sequence
                full_text = ' '.join(context[-self.config.context_window:] + [sequence])

                # Get LM score
                lm_score = self._get_lm_score(full_text, sequence)

                # Apply corpus adaptation
                if self.config.use_corpus_adaptation:
                    corpus_score = self.corpus_adapter.get_word_score(sequence.split()[-1])
                    lm_score += 0.1 * math.log(corpus_score)

                # Update beam score
                beam_scores[batch_idx, beam_idx] = (
                    self.config.neural_weight * beam_scores[batch_idx, beam_idx] +
                    self.config.lm_weight * lm_score
                )

        return beam_scores

    def _get_lm_score(self, full_text: str, target_word: str) -> float:
        """Get language model score for target word given context"""
        # Check cache first
        cache_key = f"{full_text}|{target_word}"
        if cache_key in self.lm_cache:
            self.cache_hits += 1
            return self.lm_cache[cache_key]

        self.cache_misses += 1

        if self.lm_model is None or self.tokenizer is None:
            return 0.0

        try:
            # Tokenize
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.config.max_context_length,
                truncation=True
            ).to(self.lm_model.device)

            # Get logits
            with torch.no_grad():
                outputs = self.lm_model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits

            # Get probability for target word
            target_tokens = self.tokenizer.encode(target_word, add_special_tokens=False)
            if target_tokens:
                target_idx = target_tokens[0]
                score = F.log_softmax(logits, dim=-1)[target_idx].item()
            else:
                score = -10.0  # Low score for unknown words

            # Cache result
            if len(self.lm_cache) < self.config.cache_size:
                self.lm_cache[cache_key] = score

            return score

        except Exception as e:
            logger.warning(f"Error computing LM score: {e}")
            return 0.0

    def _get_lm_hidden(self, context_text: str) -> Optional[torch.Tensor]:
        """Get language model hidden states for context"""
        if self.lm_model is None or self.tokenizer is None:
            return None

        try:
            inputs = self.tokenizer(
                context_text,
                return_tensors="pt",
                max_length=self.config.max_context_length,
                truncation=True
            ).to(self.lm_model.device)

            with torch.no_grad():
                outputs = self.lm_model(**inputs, output_hidden_states=True)
                # Use last layer hidden state, last token
                hidden = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_dim]

            return hidden

        except Exception as e:
            logger.warning(f"Error getting LM hidden states: {e}")
            return None

    def _get_corpus_scores(
        self,
        context: List[str],
        candidate_texts: List[str]
    ) -> torch.Tensor:
        """Get corpus adaptation scores"""
        batch_size = len(candidate_texts)
        vocab_size = len(candidate_texts[0]) if candidate_texts else 1

        scores = torch.zeros(batch_size, vocab_size)

        for batch_idx, candidates in enumerate(candidate_texts):
            if isinstance(candidates, str):
                candidates = [candidates]

            for vocab_idx, candidate in enumerate(candidates):
                if vocab_idx >= vocab_size:
                    break

                # Get corpus score
                words = candidate.split()
                if words:
                    word_score = self.corpus_adapter.get_word_score(words[-1])

                    # Add bigram score if context available
                    if context and len(words) > 0:
                        bigram_score = self.corpus_adapter.get_bigram_score(
                            context[-1], words[0]
                        )
                        word_score += bigram_score

                    scores[batch_idx, vocab_idx] = math.log(word_score)

        return scores

    def forward(
        self,
        neural_logits: torch.Tensor,
        context: List[str],
        candidate_texts: Optional[List[str]] = None,
        neural_hidden: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass for LM fusion.

        Args:
            neural_logits: Neural decoder logits
            context: Previous context words
            candidate_texts: Candidate text sequences
            neural_hidden: Hidden states for deep fusion
            layer_idx: Layer index for deep fusion

        Returns:
            Fused logits
        """
        if self.config.fusion_method == "shallow":
            return self.shallow_fusion(neural_logits, context, candidate_texts or [])
        elif self.config.fusion_method == "deep" and neural_hidden is not None:
            _, fused_logits = self.deep_fusion(neural_hidden, neural_logits, context, layer_idx or 0)
            return fused_logits
        elif self.config.fusion_method == "attention" and neural_hidden is not None:
            return self.attention_fusion(neural_hidden, neural_logits, context)
        else:
            return neural_logits

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.lm_cache)
        }

    def clear_cache(self):
        """Clear LM cache"""
        self.lm_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


def create_lm_fusion(config: Union[LMFusionConfig, DictConfig]) -> LanguageModelFusion:
    """Factory function to create language model fusion"""
    if isinstance(config, DictConfig):
        # Convert OmegaConf to dataclass
        config = LMFusionConfig(**config)

    return LanguageModelFusion(config)