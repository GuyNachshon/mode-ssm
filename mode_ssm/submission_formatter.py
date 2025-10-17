"""
Submission formatter for Brain-to-Text 2025 competition.
Ensures chronological ordering and proper CSV format compliance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from collections import defaultdict


logger = logging.getLogger(__name__)


@dataclass
class SubmissionRecord:
    """Single submission record"""
    session: str
    block: int
    trial: int
    prediction: str

    def __post_init__(self):
        """Validate record after initialization"""
        if self.prediction is None:
            self.prediction = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubmissionRecord':
        """Create record from dictionary"""
        return cls(
            session=data.get('session_id', data.get('session', '')),
            block=data.get('block_num', data.get('block', 0)),
            trial=data.get('trial_num', data.get('trial', 0)),
            prediction=data.get('predicted_text', data.get('prediction', ''))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary"""
        return {
            'session': self.session,
            'block': self.block,
            'trial': self.trial,
            'prediction': self.prediction
        }

    def is_valid(self) -> bool:
        """Check if record is valid"""
        if not self.session or self.session.strip() == "":
            raise ValueError("Session cannot be empty")
        if self.block <= 0:
            raise ValueError("Block must be positive")
        if self.trial <= 0:
            raise ValueError("Trial must be positive")
        if self.prediction is None:
            raise ValueError("Prediction cannot be None")
        return True

    def get_sort_key(self) -> Tuple[str, int, int]:
        """Get sorting key for chronological ordering"""
        return (self.session, self.block, self.trial)

    def __lt__(self, other: 'SubmissionRecord') -> bool:
        """Less than comparison for sorting"""
        return self.get_sort_key() < other.get_sort_key()


class SubmissionFormatter:
    """
    Formatter for competition submissions with validation and chronological ordering.

    Features:
    - Automatic chronological sorting (session → block → trial)
    - Duplicate detection and prevention
    - CSV format compliance
    - Submission validation
    - Statistics and reporting
    """

    def __init__(
        self,
        validate_format: bool = True,
        validate_order: bool = True
    ):
        """
        Initialize submission formatter.

        Args:
            validate_format: Whether to validate record format
            validate_order: Whether to validate chronological order
        """
        self.records: List[SubmissionRecord] = []
        self.validate_format = validate_format
        self.validate_order = validate_order
        self._record_keys: set = set()  # For duplicate detection

    def add_prediction(
        self,
        session: str,
        block: int,
        trial: int,
        prediction: str
    ):
        """
        Add a prediction record.

        Args:
            session: Session identifier
            block: Block number
            trial: Trial number
            prediction: Prediction text
        """
        record = SubmissionRecord(session, block, trial, prediction)

        # Validate if enabled
        if self.validate_format:
            record.is_valid()

        # Check for duplicates
        record_key = (session, block, trial)
        if record_key in self._record_keys:
            raise ValueError(
                f"Duplicate record found: session={session}, block={block}, trial={trial}"
            )

        self.records.append(record)
        self._record_keys.add(record_key)

    def add_prediction_from_dict(self, data: Dict[str, Any]):
        """Add prediction from dictionary"""
        record = SubmissionRecord.from_dict(data)

        if self.validate_format:
            record.is_valid()

        # Check for duplicates
        record_key = record.get_sort_key()
        if record_key in self._record_keys:
            raise ValueError(f"Duplicate record found: {record_key}")

        self.records.append(record)
        self._record_keys.add(record_key)

    def sort_chronologically(self):
        """Sort records in chronological order"""
        self.records.sort()

        if self.validate_order:
            self._validate_chronological_order()

    def _validate_chronological_order(self):
        """Validate that records are in chronological order"""
        prev_key = None

        for record in self.records:
            current_key = record.get_sort_key()

            if prev_key is not None and current_key <= prev_key:
                raise ValueError(
                    f"Chronological order violation: {current_key} should come after {prev_key}"
                )

            prev_key = current_key

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to pandas DataFrame in competition format (id, text)"""
        if not self.records:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['id', 'text'])

        data = [record.to_dict() for record in self.records]
        df = pd.DataFrame(data)

        # Ensure correct data types
        df['session'] = df['session'].astype(str)
        df['block'] = df['block'].astype(int)
        df['trial'] = df['trial'].astype(int)
        df['prediction'] = df['prediction'].astype(str)

        # Create competition format: id (0 to n-1) and text columns
        df['id'] = range(len(df))
        df['text'] = df['prediction']

        # Return only the required columns in correct order
        return df[['id', 'text']]

    def to_csv_string(self) -> str:
        """Convert to CSV string"""
        df = self.to_dataframe()
        return df.to_csv(index=False)

    def save_to_file(self, filepath: Union[str, Path]):
        """
        Save submission to CSV file.

        Args:
            filepath: Path to save CSV file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

        logger.info(f"Submission saved: {filepath} ({len(self.records)} records)")

    def clear(self):
        """Clear all records"""
        self.records.clear()
        self._record_keys.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get submission statistics"""
        if not self.records:
            return {
                'total_records': 0,
                'num_sessions': 0,
                'sessions': {}
            }

        # Overall stats
        stats = {
            'total_records': len(self.records),
            'num_sessions': len(set(r.session for r in self.records)),
            'sessions': defaultdict(dict)
        }

        # Per-session stats
        session_records = defaultdict(list)
        for record in self.records:
            session_records[record.session].append(record)

        for session_id, records in session_records.items():
            blocks = set(r.block for r in records)
            trials = [r.trial for r in records]

            stats['sessions'][session_id] = {
                'num_records': len(records),
                'num_blocks': len(blocks),
                'min_trial': min(trials),
                'max_trial': max(trials),
                'avg_prediction_length': np.mean([len(r.prediction) for r in records])
            }

        return dict(stats)

    def validate_submission(self) -> Tuple[bool, List[str]]:
        """
        Validate the current submission.

        Returns:
            Tuple of (is_valid, error_list)
        """
        errors = []

        if not self.records:
            errors.append("No records in submission")
            return False, errors

        # Check for required fields
        for i, record in enumerate(self.records):
            try:
                record.is_valid()
            except ValueError as e:
                errors.append(f"Record {i}: {str(e)}")

        # Check chronological order
        try:
            self._validate_chronological_order()
        except ValueError as e:
            errors.append(f"Chronological order error: {str(e)}")

        # Check for empty predictions
        empty_predictions = [
            i for i, record in enumerate(self.records)
            if not record.prediction.strip()
        ]
        if empty_predictions:
            errors.append(f"Empty predictions at records: {empty_predictions}")

        return len(errors) == 0, errors

    def __len__(self) -> int:
        """Return number of records"""
        return len(self.records)

    def __iter__(self):
        """Iterate over records"""
        return iter(self.records)


def validate_submission_format(filepath: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    Validate a CSV submission file.

    Args:
        filepath: Path to CSV file

    Returns:
        Tuple of (is_valid, error_list)
    """
    filepath = Path(filepath)
    errors = []

    # Check file exists
    if not filepath.exists():
        errors.append(f"File not found: {filepath}")
        return False, errors

    # Try to read CSV
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        errors.append(f"Cannot read CSV file: {str(e)}")
        return False, errors

    # Check required columns (competition format)
    required_columns = ['id', 'text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    # Check data types
    try:
        df['id'] = pd.to_numeric(df['id'], errors='raise').astype(int)
        df['text'] = df['text'].astype(str)
    except Exception as e:
        errors.append(f"Invalid data type in CSV: {str(e)}")

    # Check for empty predictions
    empty_mask = (df['text'].isna()) | (df['text'] == '') | (df['text'].str.strip() == '')
    if empty_mask.any():
        empty_indices = df[empty_mask].index.tolist()
        errors.append(f"Empty predictions at rows: {empty_indices}")

    # Check for duplicate IDs
    duplicates = df.duplicated(subset=['id'])
    if duplicates.any():
        duplicate_indices = df[duplicates].index.tolist()
        errors.append(f"Duplicate IDs at rows: {duplicate_indices}")

    # Check that IDs are sequential from 0 to n-1
    expected_ids = list(range(len(df)))
    actual_ids = df['id'].tolist()
    if actual_ids != expected_ids:
        errors.append(f"IDs must be sequential from 0 to {len(df)-1}, got: {actual_ids[:10]}...")

    return len(errors) == 0, errors


def ensure_chronological_order(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Check if DataFrame is in chronological order.

    Args:
        df: DataFrame with session, block, trial columns

    Returns:
        Tuple of (is_ordered, error_message)
    """
    if df.empty:
        return True, None

    # Sort by session, block, trial
    df_sorted = df.sort_values(['session', 'block', 'trial']).reset_index(drop=True)

    # Compare with original
    if not df.equals(df_sorted):
        # Find first violation
        for i in range(len(df)):
            if (df.iloc[i]['session'] != df_sorted.iloc[i]['session'] or
                df.iloc[i]['block'] != df_sorted.iloc[i]['block'] or
                df.iloc[i]['trial'] != df_sorted.iloc[i]['trial']):

                return False, (
                    f"Records not in chronological order. First violation at row {i}: "
                    f"expected {df_sorted.iloc[i].to_dict()}, "
                    f"got {df.iloc[i].to_dict()}"
                )

    return True, None


def create_submission_from_predictions(
    predictions: List[Dict[str, Any]],
    output_path: Union[str, Path],
    validate: bool = True
) -> Tuple[bool, List[str]]:
    """
    Create submission CSV from prediction list.

    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save submission
        validate: Whether to validate submission

    Returns:
        Tuple of (success, error_list)
    """
    try:
        formatter = SubmissionFormatter(
            validate_format=validate,
            validate_order=validate
        )

        # Add all predictions
        for pred in predictions:
            formatter.add_prediction_from_dict(pred)

        # Sort chronologically
        formatter.sort_chronologically()

        # Save to file
        formatter.save_to_file(output_path)

        # Final validation if requested
        if validate:
            is_valid, errors = validate_submission_format(output_path)
            if not is_valid:
                return False, errors

        return True, []

    except Exception as e:
        return False, [f"Error creating submission: {str(e)}"]