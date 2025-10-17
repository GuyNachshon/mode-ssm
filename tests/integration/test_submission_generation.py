"""
Integration tests for complete submission generation pipeline.
Tests end-to-end workflow from model predictions to valid CSV submission.
"""

import pytest
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from mode_ssm.submission_formatter import SubmissionFormatter
from mode_ssm.models.mode_ssm_model import MODESSMModel, MODESSMConfig
from datasets.brain2text import Brain2TextDataset
from tests.fixtures.synthetic_neural import SyntheticNeuralDataGenerator


class TestSubmissionGenerationPipeline:
    """Integration tests for complete submission generation"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def synthetic_data_generator(self):
        """Create synthetic data generator"""
        return SyntheticNeuralDataGenerator(
            num_channels=512,
            vocab_size=40,
            sampling_rate=50.0,
            seed=42
        )

    @pytest.fixture
    def test_dataset_file(self, temp_dir, synthetic_data_generator):
        """Create test dataset file"""
        dataset_file = temp_dir / "test_data.h5"

        # Create small test dataset with known structure
        synthetic_data_generator.create_hdf5_dataset(
            output_path=dataset_file,
            num_trials=20,
            min_seq_len=50,
            max_seq_len=200,
            sessions=['session1', 'session2'],
            blocks_per_session=2
        )

        return dataset_file

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model_config = MODESSMConfig(
            d_model=256,  # Smaller for testing
            d_state=32,
            encoder_layers=2,
            num_channels=512,
            vocab_size=40
        )

        model = MODESSMModel(model_config)
        model.eval()

        return model

    def test_end_to_end_submission_generation(
        self,
        temp_dir,
        test_dataset_file,
        mock_model
    ):
        """Test complete end-to-end submission generation pipeline"""

        # 1. Load dataset
        dataset = Brain2TextDataset(
            hdf5_path=test_dataset_file,
            cache_data=False,
            filter_quality=False  # Accept all samples for testing
        )

        assert len(dataset) > 0

        # 2. Create submission formatter
        formatter = SubmissionFormatter()

        # 3. Process dataset and generate predictions
        with torch.no_grad():
            for idx in range(len(dataset)):
                sample = dataset[idx]

                # Create batch from sample
                batch = {
                    'neural_features': sample['neural_features'].unsqueeze(0),
                    'sequence_lengths': sample['sequence_length'].unsqueeze(0)
                }

                # Run inference
                results = mock_model.inference(
                    neural_features=batch['neural_features'],
                    sequence_lengths=batch['sequence_lengths'],
                    decode_mode='greedy'
                )

                # Create prediction text (simplified)
                decoded_sequence = results['decoded_sequences'][0]
                prediction_text = ' '.join([f'phoneme_{idx}' for idx in decoded_sequence[:5]])

                # Add to formatter with proper session/block/trial info
                formatter.add_prediction(
                    session=sample['session_id'],
                    block=sample['block_num'].item(),
                    trial=sample['trial_num'].item(),
                    prediction=prediction_text
                )

        # 4. Generate CSV submission
        assert len(formatter.records) == len(dataset)

        # Sort chronologically
        formatter.sort_chronologically()

        # Convert to DataFrame
        df = formatter.to_dataframe()

        # 5. Validate submission format
        # Check required columns
        required_columns = ['session', 'block', 'trial', 'prediction']
        assert all(col in df.columns for col in required_columns)

        # Check data types
        assert df['session'].dtype == 'object'
        assert pd.api.types.is_integer_dtype(df['block'])
        assert pd.api.types.is_integer_dtype(df['trial'])
        assert df['prediction'].dtype == 'object'

        # Check no empty predictions
        assert not df['prediction'].isna().any()
        assert not (df['prediction'] == '').any()

        # Check chronological ordering
        prev_session = None
        prev_block = -1
        prev_trial = -1

        for _, row in df.iterrows():
            session = row['session']
            block = row['block']
            trial = row['trial']

            if prev_session is None or session != prev_session:
                # New session
                prev_session = session
                prev_block = block
                prev_trial = trial
            elif block != prev_block:
                # New block within same session
                assert block >= prev_block, f"Block order violation: {block} < {prev_block}"
                prev_block = block
                prev_trial = trial
            else:
                # Same block - trial should increment
                assert trial > prev_trial, f"Trial order violation: {trial} <= {prev_trial}"
                prev_trial = trial

        # 6. Save to file
        submission_file = temp_dir / "submission.csv"
        formatter.save_to_file(submission_file)

        # Verify file exists and can be read
        assert submission_file.exists()

        loaded_df = pd.read_csv(submission_file)
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == required_columns

    def test_batch_processing_consistency(
        self,
        temp_dir,
        test_dataset_file,
        mock_model
    ):
        """Test that batch processing produces consistent results"""

        dataset = Brain2TextDataset(
            hdf5_path=test_dataset_file,
            cache_data=False,
            filter_quality=False
        )

        from torch.utils.data import DataLoader
        from datasets.brain2text import collate_batch

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_batch
        )

        formatter = SubmissionFormatter()

        # Process in batches
        with torch.no_grad():
            for batch in dataloader:
                # Run batch inference
                results = mock_model.inference(
                    neural_features=batch['neural_features'],
                    sequence_lengths=batch['sequence_lengths'],
                    decode_mode='greedy'
                )

                # Process each sample in batch
                batch_size = len(batch['neural_features'])
                for i in range(batch_size):
                    decoded_sequence = results['decoded_sequences'][i]
                    prediction_text = ' '.join([f'phoneme_{idx}' for idx in decoded_sequence[:3]])

                    formatter.add_prediction(
                        session=batch['session_ids'][i],
                        block=batch['block_nums'][i].item(),
                        trial=batch['trial_nums'][i].item(),
                        prediction=prediction_text
                    )

        # Verify all samples processed
        assert len(formatter.records) == len(dataset)

        # Check that records can be sorted chronologically
        formatter.sort_chronologically()
        df = formatter.to_dataframe()

        # No duplicates
        duplicates = df.duplicated(subset=['session', 'block', 'trial'])
        assert not duplicates.any(), "Found duplicate session/block/trial combinations"

    def test_submission_statistics_and_validation(
        self,
        temp_dir,
        test_dataset_file,
        mock_model
    ):
        """Test submission statistics and validation features"""

        dataset = Brain2TextDataset(
            hdf5_path=test_dataset_file,
            cache_data=False,
            filter_quality=False
        )

        formatter = SubmissionFormatter()

        # Generate predictions
        with torch.no_grad():
            for idx in range(min(10, len(dataset))):  # Limit for testing
                sample = dataset[idx]

                batch = {
                    'neural_features': sample['neural_features'].unsqueeze(0),
                    'sequence_lengths': sample['sequence_length'].unsqueeze(0)
                }

                results = mock_model.inference(
                    neural_features=batch['neural_features'],
                    sequence_lengths=batch['sequence_lengths'],
                    decode_mode='greedy'
                )

                decoded_sequence = results['decoded_sequences'][0]
                prediction_text = f'sample_{idx}_prediction'

                formatter.add_prediction(
                    session=sample['session_id'],
                    block=sample['block_num'].item(),
                    trial=sample['trial_num'].item(),
                    prediction=prediction_text
                )

        # Get statistics
        stats = formatter.get_statistics()

        assert stats['total_records'] == min(10, len(dataset))
        assert stats['num_sessions'] > 0
        assert 'sessions' in stats

        # Check session statistics
        for session_id, session_stats in stats['sessions'].items():
            assert 'num_records' in session_stats
            assert session_stats['num_records'] > 0

        # Save and validate submission
        submission_file = temp_dir / "test_submission.csv"
        formatter.save_to_file(submission_file)

        # Basic file validation
        from mode_ssm.submission_formatter import validate_submission_format
        is_valid, errors = validate_submission_format(submission_file)

        if not is_valid:
            for error in errors:
                print(f"Validation error: {error}")

        assert is_valid, f"Submission validation failed: {errors}"

    def test_error_handling_in_pipeline(
        self,
        temp_dir,
        test_dataset_file,
        mock_model
    ):
        """Test error handling in submission generation pipeline"""

        dataset = Brain2TextDataset(
            hdf5_path=test_dataset_file,
            cache_data=False,
            filter_quality=False
        )

        formatter = SubmissionFormatter()

        # Test handling of failed model inference
        with torch.no_grad():
            for idx in range(3):
                sample = dataset[idx]

                try:
                    batch = {
                        'neural_features': sample['neural_features'].unsqueeze(0),
                        'sequence_lengths': sample['sequence_length'].unsqueeze(0)
                    }

                    results = mock_model.inference(
                        neural_features=batch['neural_features'],
                        sequence_lengths=batch['sequence_lengths'],
                        decode_mode='greedy'
                    )

                    decoded_sequence = results['decoded_sequences'][0]
                    prediction_text = f'prediction_{idx}'

                    formatter.add_prediction(
                        session=sample['session_id'],
                        block=sample['block_num'].item(),
                        trial=sample['trial_num'].item(),
                        prediction=prediction_text
                    )

                except Exception as e:
                    # Handle inference failure by adding placeholder
                    formatter.add_prediction(
                        session=sample['session_id'],
                        block=sample['block_num'].item(),
                        trial=sample['trial_num'].item(),
                        prediction='inference_failed'
                    )

        # Should have processed all samples even with some failures
        assert len(formatter.records) == 3

        # Test duplicate handling
        with pytest.raises(ValueError, match="Duplicate record"):
            sample = dataset[0]
            formatter.add_prediction(
                session=sample['session_id'],
                block=sample['block_num'].item(),
                trial=sample['trial_num'].item(),
                prediction='duplicate_attempt'
            )

    def test_memory_efficient_processing(
        self,
        temp_dir,
        test_dataset_file,
        mock_model
    ):
        """Test memory-efficient processing for large datasets"""

        dataset = Brain2TextDataset(
            hdf5_path=test_dataset_file,
            cache_data=False,  # Don't cache to test memory efficiency
            filter_quality=False
        )

        formatter = SubmissionFormatter()

        # Process samples one by one to minimize memory usage
        processed_count = 0
        batch_size = 2

        for start_idx in range(0, min(10, len(dataset)), batch_size):
            end_idx = min(start_idx + batch_size, len(dataset))

            # Create mini-batch
            mini_batch_features = []
            mini_batch_lengths = []
            mini_batch_metadata = []

            for idx in range(start_idx, end_idx):
                sample = dataset[idx]
                mini_batch_features.append(sample['neural_features'])
                mini_batch_lengths.append(sample['sequence_length'])
                mini_batch_metadata.append({
                    'session_id': sample['session_id'],
                    'block_num': sample['block_num'].item(),
                    'trial_num': sample['trial_num'].item()
                })

            # Stack batch
            batch = {
                'neural_features': torch.stack(mini_batch_features),
                'sequence_lengths': torch.stack(mini_batch_lengths)
            }

            # Process batch
            with torch.no_grad():
                results = mock_model.inference(
                    neural_features=batch['neural_features'],
                    sequence_lengths=batch['sequence_lengths'],
                    decode_mode='greedy'
                )

                # Add predictions
                for i, metadata in enumerate(mini_batch_metadata):
                    decoded_sequence = results['decoded_sequences'][i]
                    prediction_text = f'batch_prediction_{processed_count}'

                    formatter.add_prediction(
                        session=metadata['session_id'],
                        block=metadata['block_num'],
                        trial=metadata['trial_num'],
                        prediction=prediction_text
                    )

                    processed_count += 1

            # Clear batch from memory
            del batch, results, mini_batch_features

        assert processed_count == min(10, len(dataset))
        assert len(formatter.records) == processed_count

    def test_submission_format_compliance(self, temp_dir):
        """Test that generated submissions comply with competition format"""

        formatter = SubmissionFormatter()

        # Add sample data in specific format
        test_data = [
            ('session1', 1, 1, 'hello world'),
            ('session1', 1, 2, 'how are you'),
            ('session1', 2, 1, 'goodbye'),
            ('session2', 1, 1, 'thank you'),
        ]

        for session, block, trial, prediction in test_data:
            formatter.add_prediction(session, block, trial, prediction)

        # Generate CSV
        submission_file = temp_dir / "format_test.csv"
        formatter.save_to_file(submission_file)

        # Load and verify exact format
        with open(submission_file, 'r') as f:
            lines = f.readlines()

        # Check header
        assert lines[0].strip() == 'session,block,trial,prediction'

        # Check data lines
        expected_lines = [
            'session1,1,1,hello world',
            'session1,1,2,how are you',
            'session1,2,1,goodbye',
            'session2,1,1,thank you'
        ]

        for i, expected_line in enumerate(expected_lines):
            assert lines[i + 1].strip() == expected_line

        # Verify no extra whitespace or formatting issues
        df = pd.read_csv(submission_file)
        assert not df['prediction'].str.contains('\n').any()
        assert not df['prediction'].str.startswith(' ').any()
        assert not df['prediction'].str.endswith(' ').any()

    def test_large_dataset_simulation(self, temp_dir):
        """Simulate processing large dataset without actually creating one"""

        formatter = SubmissionFormatter()

        # Simulate processing 1000 samples across multiple sessions
        num_samples = 1000
        sessions = [f'session{i:03d}' for i in range(1, 11)]  # 10 sessions

        for i in range(num_samples):
            session = sessions[i % len(sessions)]
            block = (i // 50) % 10 + 1  # 10 blocks per session
            trial = i % 50 + 1  # 50 trials per block

            formatter.add_prediction(
                session=session,
                block=block,
                trial=trial,
                prediction=f'prediction_sample_{i:04d}'
            )

        # Verify all samples added
        assert len(formatter.records) == num_samples

        # Sort and validate
        formatter.sort_chronologically()
        df = formatter.to_dataframe()

        # Check no duplicates
        duplicates = df.duplicated(subset=['session', 'block', 'trial'])
        assert not duplicates.any()

        # Check chronological order
        from mode_ssm.submission_formatter import ensure_chronological_order
        is_ordered, error = ensure_chronological_order(df)
        assert is_ordered, f"Chronological order violation: {error}"

        # Save and validate
        submission_file = temp_dir / "large_submission.csv"
        formatter.save_to_file(submission_file)

        # Quick file size check
        file_size = submission_file.stat().st_size
        assert file_size > 1000  # Should be reasonably large

        # Spot check a few lines
        loaded_df = pd.read_csv(submission_file)
        assert len(loaded_df) == num_samples
        assert 'session001' in loaded_df['session'].values
        assert loaded_df['block'].max() <= 10
        assert loaded_df['trial'].max() <= 50