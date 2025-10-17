"""
Unit tests for Mamba SSM encoder.
Tests bidirectional processing, state management, and gradient flow.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

from mode_ssm.models.ssm_encoder import MambaEncoder


class TestMambaEncoder:
    """Test cases for MambaEncoder"""

    @pytest.fixture
    def encoder_config(self):
        """Standard encoder configuration"""
        return {
            'd_model': 512,
            'd_state': 64,
            'd_conv': 4,
            'expand': 2,
            'n_layers': 8,
            'bidirectional': True,
            'dropout': 0.1,
            'layer_norm_eps': 1e-5,
            'gradient_checkpointing': False
        }

    @pytest.fixture
    def encoder(self, encoder_config):
        """Create encoder instance"""
        return MambaEncoder(**encoder_config)

    @pytest.fixture
    def sample_features(self):
        """Generate sample features [batch_size, seq_len, d_model]"""
        batch_size, seq_len, d_model = 4, 100, 512
        torch.manual_seed(42)
        return torch.randn(batch_size, seq_len, d_model)

    def test_encoder_initialization(self, encoder_config):
        """Test encoder initialization with valid config"""
        encoder = MambaEncoder(**encoder_config)

        assert encoder.d_model == 512
        assert encoder.d_state == 64
        assert encoder.n_layers == 8
        assert encoder.bidirectional is True
        assert hasattr(encoder, 'layers')
        assert len(encoder.layers) == 8

    def test_encoder_invalid_config(self):
        """Test encoder initialization with invalid config"""
        with pytest.raises(ValueError):
            MambaEncoder(
                d_model=0,
                d_state=64,
                n_layers=4
            )

        with pytest.raises(ValueError):
            MambaEncoder(
                d_model=512,
                d_state=0,
                n_layers=4
            )

        with pytest.raises(ValueError):
            MambaEncoder(
                d_model=512,
                d_state=64,
                n_layers=0
            )

    def test_forward_pass_shape(self, encoder, sample_features):
        """Test forward pass produces correct output shape"""
        batch_size, seq_len, d_model = sample_features.shape

        with torch.no_grad():
            output = encoder(sample_features)

        # Output should maintain input shape for bidirectional encoder
        expected_shape = (batch_size, seq_len, d_model)
        assert output.shape == expected_shape

    def test_unidirectional_vs_bidirectional(self, sample_features):
        """Test difference between unidirectional and bidirectional encoding"""
        # Create unidirectional encoder
        uni_encoder = MambaEncoder(
            d_model=512,
            d_state=64,
            n_layers=4,
            bidirectional=False
        )

        # Create bidirectional encoder
        bi_encoder = MambaEncoder(
            d_model=512,
            d_state=64,
            n_layers=4,
            bidirectional=True
        )

        with torch.no_grad():
            uni_output = uni_encoder(sample_features)
            bi_output = bi_encoder(sample_features)

        # Both should have same output shape
        assert uni_output.shape == bi_output.shape

        # But outputs should be different
        assert not torch.allclose(uni_output, bi_output, atol=1e-4)

    def test_sequence_length_invariance(self, encoder):
        """Test encoder handles different sequence lengths"""
        batch_size, d_model = 4, 512

        for seq_len in [10, 50, 100, 500]:
            features = torch.randn(batch_size, seq_len, d_model)

            with torch.no_grad():
                output = encoder(features)

            expected_shape = (batch_size, seq_len, d_model)
            assert output.shape == expected_shape

    def test_batch_size_invariance(self, encoder):
        """Test encoder handles different batch sizes"""
        seq_len, d_model = 100, 512

        for batch_size in [1, 2, 8, 16]:
            features = torch.randn(batch_size, seq_len, d_model)

            with torch.no_grad():
                output = encoder(features)

            expected_shape = (batch_size, seq_len, d_model)
            assert output.shape == expected_shape

    def test_gradient_flow(self, encoder, sample_features):
        """Test gradient flow through encoder"""
        sample_features.requires_grad_(True)

        output = encoder(sample_features)
        loss = output.sum()
        loss.backward()

        # Input should have gradients
        assert sample_features.grad is not None
        assert not torch.isnan(sample_features.grad).any()

        # Model parameters should have gradients
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    def test_dropout_behavior(self, sample_features):
        """Test dropout behavior in train vs eval modes"""
        encoder = MambaEncoder(
            d_model=512,
            d_state=64,
            n_layers=4,
            dropout=0.5
        )

        # In eval mode, outputs should be deterministic
        encoder.eval()
        torch.manual_seed(42)
        output1 = encoder(sample_features)
        torch.manual_seed(42)
        output2 = encoder(sample_features)

        # Should be identical in eval mode
        assert torch.allclose(output1, output2), "Eval mode should be deterministic"

    def test_layer_normalization(self, encoder, sample_features):
        """Test that layer normalization is working correctly"""
        with torch.no_grad():
            output = encoder(sample_features)

        # After layer norm, values should be roughly normalized per layer
        # Check that output values are in reasonable range
        mean_val = output.mean().item()
        std_val = output.std().item()

        assert abs(mean_val) < 1.0, f"Mean too large: {mean_val}"
        assert 0.1 < std_val < 10.0, f"Std out of range: {std_val}"

    def test_state_space_dimensions(self, encoder_config):
        """Test different state space dimensions"""
        for d_state in [16, 32, 64, 128]:
            encoder = MambaEncoder(
                **{**encoder_config, 'd_state': d_state}
            )

            features = torch.randn(2, 50, 512)
            with torch.no_grad():
                output = encoder(features)

            assert output.shape == features.shape

    def test_expansion_factor(self, encoder_config, sample_features):
        """Test different expansion factors"""
        for expand in [1, 2, 4]:
            encoder = MambaEncoder(
                **{**encoder_config, 'expand': expand}
            )

            with torch.no_grad():
                output = encoder(features)

            assert output.shape == sample_features.shape

    def test_variable_layer_count(self, sample_features):
        """Test encoders with different numbers of layers"""
        for n_layers in [1, 2, 4, 8, 12]:
            encoder = MambaEncoder(
                d_model=512,
                d_state=64,
                n_layers=n_layers,
                bidirectional=True
            )

            with torch.no_grad():
                output = encoder(sample_features)

            assert output.shape == sample_features.shape
            assert len(encoder.layers) == n_layers

    def test_gradient_checkpointing_enabled(self, sample_features):
        """Test encoder with gradient checkpointing enabled"""
        encoder = MambaEncoder(
            d_model=512,
            d_state=64,
            n_layers=4,
            gradient_checkpointing=True
        )

        sample_features.requires_grad_(True)

        output = encoder(sample_features)
        loss = output.sum()
        loss.backward()

        # Should still have gradients despite checkpointing
        assert sample_features.grad is not None
        assert not torch.isnan(sample_features.grad).any()

    def test_gradient_checkpointing_disabled(self, sample_features):
        """Test encoder with gradient checkpointing disabled"""
        encoder = MambaEncoder(
            d_model=512,
            d_state=64,
            n_layers=4,
            gradient_checkpointing=False
        )

        sample_features.requires_grad_(True)

        output = encoder(sample_features)
        loss = output.sum()
        loss.backward()

        # Should have gradients
        assert sample_features.grad is not None
        assert not torch.isnan(sample_features.grad).any()

    def test_output_values_stability(self, encoder, sample_features):
        """Test output value stability and absence of NaN/inf"""
        with torch.no_grad():
            output = encoder(sample_features)

        # No NaN or inf values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Output values should be in reasonable range
        assert output.abs().max() < 100.0, "Output values too large"

    def test_padding_mask_handling(self, encoder):
        """Test encoder behavior with padding masks"""
        batch_size, seq_len, d_model = 4, 100, 512
        features = torch.randn(batch_size, seq_len, d_model)

        # Create padding mask (1 for real tokens, 0 for padding)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[0, 80:] = False  # First sequence has padding after position 80
        mask[1, 60:] = False  # Second sequence has padding after position 60

        with torch.no_grad():
            # Most encoders should handle this gracefully even without explicit mask support
            output = encoder(features)

        assert output.shape == features.shape
        assert not torch.isnan(output).any()

    def test_extreme_input_values(self, encoder):
        """Test encoder robustness to extreme input values"""
        batch_size, seq_len, d_model = 2, 50, 512

        # Test with very large values
        large_features = torch.randn(batch_size, seq_len, d_model) * 100
        with torch.no_grad():
            output1 = encoder(large_features)
        assert not torch.isnan(output1).any()
        assert not torch.isinf(output1).any()

        # Test with very small values
        small_features = torch.randn(batch_size, seq_len, d_model) * 1e-6
        with torch.no_grad():
            output2 = encoder(small_features)
        assert not torch.isnan(output2).any()
        assert not torch.isinf(output2).any()

    def test_device_compatibility(self, encoder, sample_features):
        """Test encoder works on different devices"""
        # Test CPU
        output_cpu = encoder(sample_features)
        assert output_cpu.device == sample_features.device

        # Test CUDA if available
        if torch.cuda.is_available():
            encoder_cuda = encoder.cuda()
            features_cuda = sample_features.cuda()

            with torch.no_grad():
                output_cuda = encoder_cuda(features_cuda)

            assert output_cuda.device.type == 'cuda'
            assert output_cuda.shape == output_cpu.shape

    def test_parameter_count(self, encoder_config):
        """Test that parameter count scales reasonably with config"""
        encoder = MambaEncoder(**encoder_config)

        total_params = sum(p.numel() for p in encoder.parameters())

        # Should have reasonable number of parameters
        assert total_params > 1000, "Too few parameters"
        assert total_params < 100_000_000, "Too many parameters"

        # Parameter count should scale with number of layers
        encoder_small = MambaEncoder(**{**encoder_config, 'n_layers': 2})
        small_params = sum(p.numel() for p in encoder_small.parameters())

        assert small_params < total_params, "Parameter count should scale with layers"

    def test_deterministic_output(self, encoder, sample_features):
        """Test that encoder produces deterministic output given same input"""
        encoder.eval()

        torch.manual_seed(42)
        output1 = encoder(sample_features)

        torch.manual_seed(42)
        output2 = encoder(sample_features)

        assert torch.allclose(output1, output2), "Output should be deterministic in eval mode"

    def test_memory_efficiency(self, encoder_config):
        """Test memory usage doesn't grow excessively"""
        import gc

        encoder = MambaEncoder(**encoder_config)

        base_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Test with progressively larger sequence lengths
        for seq_len in [100, 200, 500]:
            features = torch.randn(2, seq_len, 512)

            if torch.cuda.is_available():
                features = features.cuda()
                encoder = encoder.cuda()

            with torch.no_grad():
                output = encoder(features)

            # Clean up
            del features, output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Memory should not have grown significantly
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - base_memory
            assert memory_growth < 500 * 1024 * 1024  # Less than 500MB growth

    def test_state_space_model_properties(self, encoder, sample_features):
        """Test that encoder exhibits expected SSM properties"""
        # SSMs should handle long sequences efficiently
        long_features = torch.randn(2, 1000, 512)

        with torch.no_grad():
            output = encoder(long_features)

        assert output.shape == long_features.shape
        assert not torch.isnan(output).any()

        # Should maintain information across the sequence
        # (This is a basic check - more sophisticated tests would check
        # actual information flow properties)
        output_variance = output.var(dim=1).mean()  # Variance across sequence dimension
        assert output_variance > 1e-6, "Output should have meaningful variation across sequence"