"""
Phoneme vocabulary mapping for T15 Brain-to-Text dataset.
40-class vocabulary including blank and silence tokens as per competition format.
"""

from typing import Dict, List, Tuple
import torch

# 41-class phoneme vocabulary from T15 competition
LOGIT_TO_PHONEME = [
    'BLANK',    # 0: CTC blank symbol
    'AA',       # 1: as in 'father'
    'AE',       # 2: as in 'cat'
    'AH',       # 3: as in 'but'
    'AO',       # 4: as in 'law'
    'AW',       # 5: as in 'how'
    'AY',       # 6: as in 'eye'
    'B',        # 7: as in 'bee'
    'CH',       # 8: as in 'cheese'
    'D',        # 9: as in 'dee'
    'DH',       # 10: as in 'thee'
    'EH',       # 11: as in 'bet'
    'ER',       # 12: as in 'her'
    'EY',       # 13: as in 'bay'
    'F',        # 14: as in 'fee'
    'G',        # 15: as in 'go'
    'HH',       # 16: as in 'he'
    'IH',       # 17: as in 'bit'
    'IY',       # 18: as in 'beat'
    'JH',       # 19: as in 'jee'
    'K',        # 20: as in 'key'
    'L',        # 21: as in 'lee'
    'M',        # 22: as in 'me'
    'N',        # 23: as in 'knee'
    'NG',       # 24: as in 'sing'
    'OW',       # 25: as in 'go'
    'OY',       # 26: as in 'toy'
    'P',        # 27: as in 'pee'
    'R',        # 28: as in 'red'
    'S',        # 29: as in 'see'
    'SH',       # 30: as in 'she'
    'T',        # 31: as in 'tea'
    'TH',       # 32: as in 'theta'
    'UH',       # 33: as in 'book'
    'UW',       # 34: as in 'boot'
    'V',        # 35: as in 'vee'
    'W',        # 36: as in 'we'
    'Y',        # 37: as in 'yes'
    'Z',        # 38: as in 'zee'
    'ZH',       # 39: as in 'measure' / silence token
    'WB',       # 40: word boundary marker
]

# Reverse mapping for phoneme to index
PHONEME_TO_LOGIT = {phoneme: idx for idx, phoneme in enumerate(LOGIT_TO_PHONEME)}

# Special token indices
BLANK_TOKEN_IDX = 0
SILENCE_TOKEN_IDX = 39
WORD_BOUNDARY_IDX = 40
VOCAB_SIZE = len(LOGIT_TO_PHONEME)

# Phoneme categories for analysis
VOWELS = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
    'IH', 'IY', 'OW', 'OY', 'UH', 'UW'
}

CONSONANTS = {
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L',
    'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V',
    'W', 'Y', 'Z', 'ZH'
}

FRICATIVES = {'F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH'}
PLOSIVES = {'P', 'B', 'T', 'D', 'K', 'G'}
NASALS = {'M', 'N', 'NG'}
LIQUIDS = {'L', 'R'}
GLIDES = {'W', 'Y'}


class PhonemeVocabulary:
    """Utility class for phoneme vocabulary operations"""

    def __init__(self):
        self.logit_to_phoneme = LOGIT_TO_PHONEME
        self.phoneme_to_logit = PHONEME_TO_LOGIT
        self.vocab_size = VOCAB_SIZE
        self.blank_idx = BLANK_TOKEN_IDX
        self.silence_idx = SILENCE_TOKEN_IDX

    def encode_phoneme_sequence(self, phonemes: List[str]) -> torch.Tensor:
        """Convert phoneme strings to tensor of indices

        Args:
            phonemes: List of phoneme strings

        Returns:
            Long tensor of phoneme indices
        """
        indices = []
        for phoneme in phonemes:
            if phoneme in self.phoneme_to_logit:
                indices.append(self.phoneme_to_logit[phoneme])
            else:
                # Unknown phoneme -> use silence token
                indices.append(self.silence_idx)
        return torch.tensor(indices, dtype=torch.long)

    def decode_phoneme_sequence(
        self,
        indices: torch.Tensor,
        remove_blanks: bool = True,
        remove_repeats: bool = True
    ) -> List[str]:
        """Convert tensor of indices to phoneme strings

        Args:
            indices: Tensor of phoneme indices
            remove_blanks: Whether to remove blank tokens
            remove_repeats: Whether to remove repeated phonemes (CTC decoding)

        Returns:
            List of phoneme strings
        """
        indices = indices.cpu().numpy() if isinstance(indices, torch.Tensor) else indices
        phonemes = []

        prev_idx = None
        for idx in indices:
            if idx < 0 or idx >= self.vocab_size:
                continue

            # Skip blanks if requested
            if remove_blanks and idx == self.blank_idx:
                prev_idx = idx
                continue

            # Skip repeats if requested (for CTC)
            if remove_repeats and idx == prev_idx:
                continue

            phonemes.append(self.logit_to_phoneme[idx])
            prev_idx = idx

        return phonemes

    def phonemes_to_text(self, phonemes: List[str]) -> str:
        """Convert phoneme sequence to readable text approximation

        Args:
            phonemes: List of phoneme strings

        Returns:
            Approximate text representation
        """
        # This is a simplified phoneme-to-text conversion
        # In practice, would use a proper G2P model in reverse

        # Basic phoneme to letter approximations
        phoneme_to_letter = {
            'AA': 'a', 'AE': 'a', 'AH': 'a', 'AO': 'o', 'AW': 'ow', 'AY': 'i',
            'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'th', 'EH': 'e', 'ER': 'er',
            'EY': 'ay', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'i', 'IY': 'ee',
            'JH': 'j', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ng',
            'OW': 'o', 'OY': 'oy', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'sh',
            'T': 't', 'TH': 'th', 'UH': 'u', 'UW': 'oo', 'V': 'v', 'W': 'w',
            'Y': 'y', 'Z': 'z', 'ZH': 'zh', 'BLANK': '', 'WB': ' '
        }

        letters = []
        for phoneme in phonemes:
            if phoneme in phoneme_to_letter:
                letters.append(phoneme_to_letter[phoneme])
            elif phoneme == 'WB':  # Word boundary marker
                letters.append(' ')

        return ''.join(letters).strip()

    def get_phoneme_features(self, phoneme: str) -> Dict[str, bool]:
        """Get phonetic features for a phoneme

        Args:
            phoneme: Phoneme string

        Returns:
            Dictionary of phonetic features
        """
        features = {
            'is_vowel': phoneme in VOWELS,
            'is_consonant': phoneme in CONSONANTS,
            'is_fricative': phoneme in FRICATIVES,
            'is_plosive': phoneme in PLOSIVES,
            'is_nasal': phoneme in NASALS,
            'is_liquid': phoneme in LIQUIDS,
            'is_glide': phoneme in GLIDES,
            'is_special': phoneme in {'BLANK', 'ZH', 'WB'}  # Special tokens
        }
        return features

    def compute_phoneme_statistics(self, phoneme_sequences: List[List[str]]) -> Dict:
        """Compute statistics over phoneme sequences

        Args:
            phoneme_sequences: List of phoneme sequences

        Returns:
            Dictionary with phoneme statistics
        """
        phoneme_counts = {phoneme: 0 for phoneme in self.logit_to_phoneme}
        total_phonemes = 0
        sequence_lengths = []

        for sequence in phoneme_sequences:
            sequence_lengths.append(len(sequence))
            for phoneme in sequence:
                if phoneme in phoneme_counts:
                    phoneme_counts[phoneme] += 1
                    total_phonemes += 1

        # Compute frequencies
        phoneme_freqs = {
            phoneme: count / max(1, total_phonemes)
            for phoneme, count in phoneme_counts.items()
        }

        return {
            'phoneme_counts': phoneme_counts,
            'phoneme_frequencies': phoneme_freqs,
            'total_phonemes': total_phonemes,
            'avg_sequence_length': sum(sequence_lengths) / len(sequence_lengths),
            'vocab_coverage': sum(1 for count in phoneme_counts.values() if count > 0)
        }


def create_phoneme_vocab() -> PhonemeVocabulary:
    """Create phoneme vocabulary instance"""
    return PhonemeVocabulary()


def validate_phoneme_sequence(sequence: List[str]) -> Tuple[bool, List[str]]:
    """Validate phoneme sequence and return errors

    Args:
        sequence: List of phoneme strings

    Returns:
        (is_valid, list_of_errors)
    """
    vocab = create_phoneme_vocab()
    errors = []

    for i, phoneme in enumerate(sequence):
        if phoneme not in vocab.phoneme_to_logit:
            errors.append(f"Unknown phoneme '{phoneme}' at position {i}")

    if len(sequence) == 0:
        errors.append("Empty phoneme sequence")

    return len(errors) == 0, errors


def ctc_decode_predictions(
    logits: torch.Tensor,
    sequence_lengths: torch.Tensor,
    blank_idx: int = BLANK_TOKEN_IDX
) -> List[List[str]]:
    """CTC decode logits to phoneme sequences

    Args:
        logits: [B, T, V] logit tensor
        sequence_lengths: [B] sequence lengths
        blank_idx: Blank token index

    Returns:
        List of decoded phoneme sequences for each batch item
    """
    vocab = create_phoneme_vocab()
    predictions = torch.argmax(logits, dim=-1)  # [B, T]

    decoded_sequences = []
    for i in range(predictions.shape[0]):
        seq_len = sequence_lengths[i].item()
        pred_seq = predictions[i, :seq_len]

        # CTC decoding: remove blanks and consecutive duplicates
        phonemes = vocab.decode_phoneme_sequence(
            pred_seq,
            remove_blanks=True,
            remove_repeats=True
        )
        decoded_sequences.append(phonemes)

    return decoded_sequences