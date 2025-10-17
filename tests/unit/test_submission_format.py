"""
Unit tests for CSV submission formatting and validation.
Tests chronological ordering, format compliance, and data integrity.
"""

import pytest
import pandas as pd
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from mode_ssm.submission_formatter import (
    SubmissionFormatter,
    SubmissionRecord,
    validate_submission_format,
    ensure_chronological_order
)


class TestSubmissionRecord:
    """Test cases for SubmissionRecord"""

    def test_submission_record_creation(self):
        """Test creating a submission record"""
        record = SubmissionRecord(
            session="session1",
            block=1,
            trial=5,
            prediction="hello world"
        )

        assert record.session == "session1"
        assert record.block == 1
        assert record.trial == 5
        assert record.prediction == "hello world"

    def test_submission_record_from_dict(self):
        """Test creating record from dictionary"""
        data = {
            'session': 'session2',
            'block': 2,
            'trial': 10,
            'prediction': 'test prediction'
        }

        record = SubmissionRecord.from_dict(data)

        assert record.session == 'session2'
        assert record.block == 2
        assert record.trial == 10
        assert record.prediction == 'test prediction'

    def test_submission_record_to_dict(self):
        """Test converting record to dictionary"""
        record = SubmissionRecord("sess1", 1, 1, "pred")
        result = record.to_dict()

        expected = {
            'session': 'sess1',
            'block': 1,
            'trial': 1,
            'prediction': 'pred'
        }

        assert result == expected

    def test_submission_record_validation(self):
        """Test submission record validation"""
        # Valid record
        valid_record = SubmissionRecord("session1", 1, 1, "hello")
        assert valid_record.is_valid()

        # Invalid records
        with pytest.raises(ValueError):
            SubmissionRecord("", 1, 1, "hello")  # Empty session

        with pytest.raises(ValueError):
            SubmissionRecord("session1", 0, 1, "hello")  # Block <= 0

        with pytest.raises(ValueError):
            SubmissionRecord("session1", 1, 0, "hello")  # Trial <= 0

        with pytest.raises(ValueError):
            SubmissionRecord("session1", 1, 1, "")  # Empty prediction


class TestSubmissionFormatter:
    """Test cases for SubmissionFormatter"""

    @pytest.fixture
    def formatter(self):
        """Create submission formatter instance"""
        return SubmissionFormatter()

    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction data"""
        return [
            {
                'session_id': 'session1',
                'block_num': 1,
                'trial_num': 1,
                'predicted_text': 'hello world'
            },
            {
                'session_id': 'session1',
                'block_num': 1,
                'trial_num': 2,
                'predicted_text': 'how are you'
            },
            {
                'session_id': 'session2',
                'block_num': 1,
                'trial_num': 1,
                'predicted_text': 'goodbye'
            }
        ]

    def test_formatter_initialization(self, formatter):
        """Test formatter initialization"""
        assert formatter.records == []
        assert formatter.validate_order is True
        assert formatter.validate_format is True

    def test_add_prediction_record(self, formatter):
        """Test adding prediction records"""
        formatter.add_prediction(
            session='session1',
            block=1,
            trial=1,
            prediction='test'
        )

        assert len(formatter.records) == 1
        record = formatter.records[0]
        assert record.session == 'session1'
        assert record.block == 1
        assert record.trial == 1
        assert record.prediction == 'test'

    def test_add_prediction_from_dict(self, formatter, sample_predictions):
        """Test adding predictions from dictionary"""
        for pred in sample_predictions:
            formatter.add_prediction_from_dict(pred)

        assert len(formatter.records) == 3

        # Check first record
        record = formatter.records[0]
        assert record.session == 'session1'
        assert record.block == 1
        assert record.trial == 1
        assert record.prediction == 'hello world'

    def test_chronological_sorting(self, formatter):
        """Test chronological sorting of records"""
        # Add records out of order
        formatter.add_prediction('session1', 2, 1, 'pred1')
        formatter.add_prediction('session1', 1, 2, 'pred2')
        formatter.add_prediction('session1', 1, 1, 'pred3')

        # Sort records
        formatter.sort_chronologically()

        # Check order
        records = formatter.records
        assert len(records) == 3

        # Should be sorted by session, block, trial
        assert (records[0].session, records[0].block, records[0].trial) == ('session1', 1, 1)
        assert (records[1].session, records[1].block, records[1].trial) == ('session1', 1, 2)
        assert (records[2].session, records[2].block, records[2].trial) == ('session1', 2, 1)

    def test_multi_session_sorting(self, formatter):
        """Test sorting across multiple sessions"""
        # Add records for multiple sessions
        formatter.add_prediction('session2', 1, 1, 'pred1')
        formatter.add_prediction('session1', 2, 1, 'pred2')
        formatter.add_prediction('session1', 1, 1, 'pred3')

        formatter.sort_chronologically()

        records = formatter.records
        sessions = [r.session for r in records]

        # Sessions should be sorted alphabetically
        assert sessions == ['session1', 'session1', 'session2']

    def test_duplicate_detection(self, formatter):
        """Test detection of duplicate records"""
        formatter.add_prediction('session1', 1, 1, 'pred1')

        # Adding duplicate should raise error
        with pytest.raises(ValueError, match="Duplicate record"):
            formatter.add_prediction('session1', 1, 1, 'pred2')

    def test_to_dataframe(self, formatter, sample_predictions):
        """Test conversion to pandas DataFrame"""
        for pred in sample_predictions:
            formatter.add_prediction_from_dict(pred)

        df = formatter.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['session', 'block', 'trial', 'prediction']

        # Check data types
        assert df['session'].dtype == 'object'
        assert df['block'].dtype == 'int64'
        assert df['trial'].dtype == 'int64'
        assert df['prediction'].dtype == 'object'

    def test_to_csv_string(self, formatter, sample_predictions):
        """Test conversion to CSV string"""
        for pred in sample_predictions:
            formatter.add_prediction_from_dict(pred)

        csv_string = formatter.to_csv_string()

        # Check that it contains headers
        assert 'session,block,trial,prediction' in csv_string

        # Check that it contains data
        assert 'session1,1,1,hello world' in csv_string
        assert 'session2,1,1,goodbye' in csv_string

        # Should not have index
        assert csv_string.count('\n') == 4  # header + 3 data rows + final newline

    def test_save_to_file(self, formatter, sample_predictions):
        """Test saving to CSV file"""
        for pred in sample_predictions:
            formatter.add_prediction_from_dict(pred)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = Path(f.name)

        try:
            formatter.save_to_file(temp_path)

            # Verify file was created and has correct content
            assert temp_path.exists()

            df_loaded = pd.read_csv(temp_path)
            assert len(df_loaded) == 3
            assert list(df_loaded.columns) == ['session', 'block', 'trial', 'prediction']

        finally:
            temp_path.unlink(missing_ok=True)

    def test_validation_enabled(self, formatter):
        """Test validation when enabled"""
        formatter.validate_format = True
        formatter.validate_order = True

        # Valid records should work
        formatter.add_prediction('session1', 1, 1, 'test')

        # Invalid records should fail
        with pytest.raises(ValueError):
            formatter.add_prediction('', 1, 1, 'test')  # Empty session

    def test_validation_disabled(self):
        """Test validation when disabled"""
        formatter = SubmissionFormatter(validate_format=False, validate_order=False)

        # Should allow invalid records when validation is disabled
        formatter.add_prediction('', 1, 1, 'test')  # Empty session - normally invalid
        assert len(formatter.records) == 1

    def test_clear_records(self, formatter, sample_predictions):
        """Test clearing all records"""
        for pred in sample_predictions:
            formatter.add_prediction_from_dict(pred)

        assert len(formatter.records) == 3

        formatter.clear()
        assert len(formatter.records) == 0

    def test_get_statistics(self, formatter, sample_predictions):
        """Test getting submission statistics"""
        for pred in sample_predictions:
            formatter.add_prediction_from_dict(pred)

        stats = formatter.get_statistics()

        assert stats['total_records'] == 3
        assert stats['num_sessions'] == 2
        assert 'session1' in stats['sessions']
        assert 'session2' in stats['sessions']
        assert stats['sessions']['session1']['num_records'] == 2
        assert stats['sessions']['session2']['num_records'] == 1

    def test_empty_prediction_handling(self, formatter):
        """Test handling of empty or None predictions"""
        # None prediction should be converted to empty string
        formatter.add_prediction('session1', 1, 1, None)
        assert formatter.records[0].prediction == ''

        # Empty string should be allowed if validation disabled
        formatter_no_val = SubmissionFormatter(validate_format=False)
        formatter_no_val.add_prediction('session1', 1, 1, '')
        assert formatter_no_val.records[0].prediction == ''


class TestSubmissionValidation:
    """Test cases for submission validation functions"""

    @pytest.fixture
    def valid_csv_content(self):
        """Valid CSV content for testing"""
        return """session,block,trial,prediction
session1,1,1,hello world
session1,1,2,how are you
session2,1,1,goodbye
"""

    @pytest.fixture
    def invalid_csv_content(self):
        """Invalid CSV content for testing"""
        return """session,block,trial,prediction
session1,1,2,hello world
session1,1,1,how are you
session2,1,1,goodbye
"""

    def test_validate_submission_format_valid(self, valid_csv_content):
        """Test validation of valid CSV format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(valid_csv_content)
            temp_path = Path(f.name)

        try:
            is_valid, errors = validate_submission_format(temp_path)
            assert is_valid is True
            assert len(errors) == 0

        finally:
            temp_path.unlink(missing_ok=True)

    def test_validate_submission_format_invalid(self, invalid_csv_content):
        """Test validation of invalid CSV format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(invalid_csv_content)
            temp_path = Path(f.name)

        try:
            is_valid, errors = validate_submission_format(temp_path)
            assert is_valid is False
            assert len(errors) > 0
            assert any('chronological order' in error for error in errors)

        finally:
            temp_path.unlink(missing_ok=True)

    def test_validate_missing_columns(self):
        """Test validation with missing columns"""
        csv_content = """session,block,prediction
session1,1,hello world
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        try:
            is_valid, errors = validate_submission_format(temp_path)
            assert is_valid is False
            assert any('Missing required columns' in error for error in errors)

        finally:
            temp_path.unlink(missing_ok=True)

    def test_validate_empty_predictions(self):
        """Test validation with empty predictions"""
        csv_content = """session,block,trial,prediction
session1,1,1,hello world
session1,1,2,
session2,1,1,goodbye
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        try:
            is_valid, errors = validate_submission_format(temp_path)
            assert is_valid is False
            assert any('Empty prediction' in error for error in errors)

        finally:
            temp_path.unlink(missing_ok=True)

    def test_ensure_chronological_order_valid(self):
        """Test chronological order validation with valid data"""
        df = pd.DataFrame({
            'session': ['session1', 'session1', 'session2'],
            'block': [1, 1, 1],
            'trial': [1, 2, 1],
            'prediction': ['a', 'b', 'c']
        })

        is_ordered, error = ensure_chronological_order(df)
        assert is_ordered is True
        assert error is None

    def test_ensure_chronological_order_invalid(self):
        """Test chronological order validation with invalid data"""
        df = pd.DataFrame({
            'session': ['session1', 'session1', 'session2'],
            'block': [1, 1, 1],
            'trial': [2, 1, 1],  # Trial 2 before trial 1
            'prediction': ['a', 'b', 'c']
        })

        is_ordered, error = ensure_chronological_order(df)
        assert is_ordered is False
        assert error is not None
        assert 'chronological order' in error

    def test_validate_nonexistent_file(self):
        """Test validation with non-existent file"""
        fake_path = Path('/nonexistent/file.csv')

        is_valid, errors = validate_submission_format(fake_path)
        assert is_valid is False
        assert any('File not found' in error for error in errors)

    def test_validate_invalid_file_format(self):
        """Test validation with non-CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("not a csv file")
            temp_path = Path(f.name)

        try:
            is_valid, errors = validate_submission_format(temp_path)
            assert is_valid is False
            assert len(errors) > 0

        finally:
            temp_path.unlink(missing_ok=True)

    def test_validate_duplicate_entries(self):
        """Test validation with duplicate session/block/trial combinations"""
        csv_content = """session,block,trial,prediction
session1,1,1,hello world
session1,1,1,duplicate entry
session2,1,1,goodbye
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        try:
            is_valid, errors = validate_submission_format(temp_path)
            assert is_valid is False
            assert any('Duplicate entry' in error for error in errors)

        finally:
            temp_path.unlink(missing_ok=True)

    def test_validate_data_types(self):
        """Test validation of data types"""
        csv_content = """session,block,trial,prediction
session1,invalid_block,1,hello world
session1,1,2,how are you
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        try:
            is_valid, errors = validate_submission_format(temp_path)
            assert is_valid is False
            assert any('Invalid data type' in error for error in errors)

        finally:
            temp_path.unlink(missing_ok=True)