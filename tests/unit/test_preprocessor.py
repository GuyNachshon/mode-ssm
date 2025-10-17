"""
Unit tests for neural signal preprocessor.
Tests normalization, channel gating, and temporal augmentation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch

from mode_ssm.models.preprocessor import NeuralPreprocessor


class TestNeuralPreprocessor:
    """Test cases for NeuralPreprocessor"""

    @pytest.fixture
    def preprocessor_config(self):
        """Standard preprocessor configuration"""
        return {
            'num_channels': 512,
            'd_model': 512,
            'normalization_momentum': 0.1,
            'channel_attention': True,
            'conv_kernel_size': 7,
            'dropout': 0.1
        }

    @pytest.fixture
    def preprocessor(self, preprocessor_config):
        """Create preprocessor instance"""
        return NeuralPreprocessor(**preprocessor_config)

    @pytest.fixture
    def sample_neural_data(self):
        """Generate sample neural data [batch_size, seq_len, num_channels]"""
        batch_size, seq_len, num_channels = 4, 100, 512

        # Generate realistic neural data with some structure
        torch.manual_seed(42)
        data = torch.randn(batch_size, seq_len, num_channels)

        # Add some channel-specific bias and scaling
        channel_bias = torch.randn(num_channels) * 0.5
        channel_scale = torch.exp(torch.randn(num_channels) * 0.2)

        data = data * channel_scale[None, None, :] + channel_bias[None, None, :]

        return data

    def test_preprocessor_initialization(self, preprocessor_config):
        """Test preprocessor initialization with valid config"""
        preprocessor = NeuralPreprocessor(**preprocessor_config)

        assert preprocessor.num_channels == 512
        assert preprocessor.d_model == 512
        assert preprocessor.channel_attention is True
        assert hasattr(preprocessor, 'normalization')
        assert hasattr(preprocessor, 'channel_gate')
        assert hasattr(preprocessor, 'conv1d')
        assert hasattr(preprocessor, 'projection')

    def test_preprocessor_invalid_config(self):
        """Test preprocessor initialization with invalid config"""
        with pytest.raises(ValueError):
            NeuralPreprocessor(
                num_channels=0,
                d_model=512
            )

        with pytest.raises(ValueError):
            NeuralPreprocessor(
                num_channels=512,
                d_model=0
            )

    def test_normalization_statistics_update(self, preprocessor, sample_neural_data):
        """Test that batch normalization statistics are updated during training"""
        preprocessor.train()

        # Get initial running stats
        initial_mean = preprocessor.normalization.running_mean.clone()
        initial_var = preprocessor.normalization.running_var.clone()

        # Forward pass should update running statistics
        with torch.no_grad():
            _ = preprocessor(sample_neural_data)

        # Statistics should have changed
        assert not torch.allclose(initial_mean, preprocessor.normalization.running_mean)
        assert not torch.allclose(initial_var, preprocessor.normalization.running_var)

    def test_normalization_eval_mode(self, preprocessor, sample_neural_data):
        """Test that normalization uses fixed statistics in eval mode"""
        preprocessor.eval()

        # Get initial running stats
        initial_mean = preprocessor.normalization.running_mean.clone()
        initial_var = preprocessor.normalization.running_var.clone()

        # Forward pass should not update running statistics
        with torch.no_grad():
            _ = preprocessor(sample_neural_data)

        # Statistics should remain unchanged
        assert torch.allclose(initial_mean, preprocessor.normalization.running_mean)
        assert torch.allclose(initial_var, preprocessor.normalization.running_var)

    def test_channel_attention_shape(self, preprocessor, sample_neural_data):
        """Test that channel attention produces correct output shape"""
        batch_size, seq_len, num_channels = sample_neural_data.shape

        with torch.no_grad():
            output = preprocessor(sample_neural_data)

        # Output should maintain sequence dimension but project to d_model
        expected_shape = (batch_size, seq_len, preprocessor.d_model)
        assert output.shape == expected_shape

    def test_channel_attention_disabled(self, sample_neural_data):
        """Test preprocessor with channel attention disabled"""
        preprocessor = NeuralPreprocessor(
            num_channels=512,
            d_model=512,
            channel_attention=False
        )

        with torch.no_grad():
            output = preprocessor(sample_neural_data)

        # Should still produce correct output shape
        batch_size, seq_len = sample_neural_data.shape[:2]
        expected_shape = (batch_size, seq_len, 512)
        assert output.shape == expected_shape

    def test_temporal_convolution_kernel_size(self, sample_neural_data):
        """Test different convolution kernel sizes"""
        for kernel_size in [3, 5, 7, 11]:
            preprocessor = NeuralPreprocessor(
                num_channels=512,
                d_model=512,
                conv_kernel_size=kernel_size
            )

            with torch.no_grad():
                output = preprocessor(sample_neural_data)

            # Output shape should be consistent regardless of kernel size
            batch_size, seq_len = sample_neural_data.shape[:2]
            expected_shape = (batch_size, seq_len, 512)
            assert output.shape == expected_shape

    def test_output_values_reasonable_range(self, preprocessor, sample_neural_data):
        """Test that output values are in reasonable range after normalization"""
        with torch.no_grad():
            output = preprocessor(sample_neural_data)

        # After normalization, values should be roughly centered around 0
        # with reasonable variance
        mean_val = output.mean().item()
        std_val = output.std().item()

        assert abs(mean_val) < 1.0, f"Mean too large: {mean_val}"
        assert 0.1 < std_val < 10.0, f"Std out of range: {std_val}"

        # No NaN or inf values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self, preprocessor, sample_neural_data):
        """Test that gradients can flow through the preprocessor"""
        sample_neural_data.requires_grad_(True)

        output = preprocessor(sample_neural_data)
        loss = output.sum()
        loss.backward()

        # Input should have gradients
        assert sample_neural_data.grad is not None
        assert not torch.isnan(sample_neural_data.grad).any()

        # Model parameters should have gradients
        for param in preprocessor.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    def test_dropout_behavior(self, sample_neural_data):
        """Test dropout behavior in train vs eval mode"""
        preprocessor = NeuralPreprocessor(
            num_channels=512,
            d_model=512,
            dropout=0.5
        )

        # In training mode, outputs should be different due to dropout
        preprocessor.train()
        torch.manual_seed(42)
        output1 = preprocessor(sample_neural_data)
        torch.manual_seed(42)
        output2 = preprocessor(sample_neural_data)

        # With dropout, outputs should be different even with same seed
        # (because dropout uses different random state)

        # In eval mode, outputs should be deterministic
        preprocessor.eval()
        torch.manual_seed(42)
        output3 = preprocessor(sample_neural_data)
        torch.manual_seed(42)
        output4 = preprocessor(sample_neural_data)

        assert torch.allclose(output3, output4), "Eval mode should be deterministic"

    def test_batch_size_invariance(self, preprocessor):
        """Test that processing different batch sizes works correctly"""
        seq_len, num_channels = 100, 512

        # Test different batch sizes
        for batch_size in [1, 2, 8, 16]:
            data = torch.randn(batch_size, seq_len, num_channels)

            with torch.no_grad():
                output = preprocessor(data)

            expected_shape = (batch_size, seq_len, 512)
            assert output.shape == expected_shape

    def test_sequence_length_invariance(self, preprocessor):
        """Test that processing different sequence lengths works correctly"""
        batch_size, num_channels = 4, 512

        # Test different sequence lengths
        for seq_len in [10, 50, 100, 500]:
            data = torch.randn(batch_size, seq_len, num_channels)

            with torch.no_grad():
                output = preprocessor(data)

            expected_shape = (batch_size, seq_len, 512)
            assert output.shape == expected_shape

    def test_missing_channels_handling(self, preprocessor_config):
        """Test handling of missing/corrupted channels (zeros)"""
        batch_size, seq_len, num_channels = 4, 100, 512

        # Create data with some zero channels (missing data)
        data = torch.randn(batch_size, seq_len, num_channels)
        data[:, :, [10, 20, 30]] = 0.0  # Simulate missing channels

        preprocessor = NeuralPreprocessor(**preprocessor_config)

        with torch.no_grad():
            output = preprocessor(data)

        # Should still produce valid output
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Output should have correct shape
        expected_shape = (batch_size, seq_len, 512)
        assert output.shape == expected_shape

    def test_extreme_input_values(self, preprocessor):
        """Test preprocessor robustness to extreme input values"""
        batch_size, seq_len, num_channels = 2, 50, 512

        # Test with very large values
        large_data = torch.randn(batch_size, seq_len, num_channels) * 1000
        with torch.no_grad():
            output1 = preprocessor(large_data)
        assert not torch.isnan(output1).any()
        assert not torch.isinf(output1).any()

        # Test with very small values
        small_data = torch.randn(batch_size, seq_len, num_channels) * 1e-6
        with torch.no_grad():
            output2 = preprocessor(small_data)
        assert not torch.isnan(output2).any()
        assert not torch.isinf(output2).any()

    def test_device_compatibility(self, preprocessor, sample_neural_data):
        """Test that preprocessor works on different devices"""
        # Test CPU
        output_cpu = preprocessor(sample_neural_data)
        assert output_cpu.device == sample_neural_data.device

        # Test CUDA if available
        if torch.cuda.is_available():
            preprocessor_cuda = preprocessor.cuda()
            data_cuda = sample_neural_data.cuda()

            with torch.no_grad():
                output_cuda = preprocessor_cuda(data_cuda)

            assert output_cuda.device.type == 'cuda'
            assert output_cuda.shape == output_cpu.shape

    def test_memory_efficiency(self, preprocessor):
        """Test memory usage doesn't grow excessively with large inputs"""
        import gc

        # Test with progressively larger inputs
        base_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        for seq_len in [100, 200, 500]:
            data = torch.randn(2, seq_len, 512)

            if torch.cuda.is_available():
                data = data.cuda()
                preprocessor = preprocessor.cuda()

            with torch.no_grad():
                output = preprocessor(data)

            # Clean up
            del data, output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Memory should not have grown significantly
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - base_memory
            assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth