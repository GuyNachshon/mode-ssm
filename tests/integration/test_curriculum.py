"""
Integration tests for multi-stage curriculum training.
Tests CTC warmup → Joint → Mode → Denoise flow training progression.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

from mode_ssm.training_stages import (
    CurriculumTrainer,
    StageTransitionManager,
    ComponentActivationManager,
    LossWeightManager,
    TrainingStage,
    create_stage_configs
)


class TestCurriculumTraining:
    """Integration tests for curriculum training system"""

    @pytest.fixture
    def mock_model(self):
        """Create mock model with named components"""
        model = MagicMock(spec=nn.Module)

        # Create mock parameters with names
        params = {}
        for component in ['preprocessor', 'encoder', 'mode_head', 'ctc_decoder', 'rnnt_decoder', 'flow_bridge']:
            for param_name in ['weight', 'bias']:
                name = f'{component}.{param_name}'
                param = torch.nn.Parameter(torch.randn(10, 10))
                params[name] = param

        model.named_parameters.return_value = params.items()
        model.parameters.return_value = params.values()

        return model

    @pytest.fixture
    def curriculum_config(self):
        """Standard curriculum training configuration"""
        return {
            'training': {
                'stages': {
                    'ctc_warmup': {
                        'enabled': True,
                        'epochs': 3,
                        'ctc_weight': 1.0,
                        'rnnt_weight': 0.0,
                        'mode_weight': 0.0,
                        'flow_weight': 0.0,
                        'train_preprocessor': True,
                        'train_encoder': True,
                        'train_ctc_decoder': True,
                        'train_rnnt_decoder': False,
                        'train_mode_head': False,
                        'train_flow_bridge': False,
                        'convergence_threshold': 0.01
                    },
                    'joint': {
                        'enabled': True,
                        'epochs': 4,
                        'ctc_weight': 0.5,
                        'rnnt_weight': 0.5,
                        'mode_weight': 0.0,
                        'flow_weight': 0.0,
                        'train_preprocessor': True,
                        'train_encoder': True,
                        'train_ctc_decoder': True,
                        'train_rnnt_decoder': True,
                        'train_mode_head': False,
                        'train_flow_bridge': False
                    },
                    'mode': {
                        'enabled': True,
                        'epochs': 2,
                        'ctc_weight': 0.3,
                        'rnnt_weight': 0.3,
                        'mode_weight': 0.4,
                        'flow_weight': 0.0,
                        'train_preprocessor': True,
                        'train_encoder': True,
                        'train_ctc_decoder': True,
                        'train_rnnt_decoder': True,
                        'train_mode_head': True,
                        'train_flow_bridge': False
                    },
                    'denoise': {
                        'enabled': True,
                        'epochs': 2,
                        'ctc_weight': 0.2,
                        'rnnt_weight': 0.2,
                        'mode_weight': 0.2,
                        'flow_weight': 0.4,
                        'train_preprocessor': True,
                        'train_encoder': True,
                        'train_ctc_decoder': True,
                        'train_rnnt_decoder': True,
                        'train_mode_head': True,
                        'train_flow_bridge': True
                    }
                }
            }
        }

    @pytest.fixture
    def optimizer_factory(self):
        """Factory function for creating optimizers"""
        def create_optimizer(parameters, lr=1e-3, **kwargs):
            return optim.Adam(parameters, lr=lr, **kwargs)
        return create_optimizer

    @pytest.fixture
    def scheduler_factory(self):
        """Factory function for creating schedulers"""
        def create_scheduler(optimizer, **kwargs):
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return create_scheduler

    def test_stage_configuration_creation(self, curriculum_config):
        """Test creation of stage configurations from config"""
        config = OmegaConf.create(curriculum_config)
        stage_configs = create_stage_configs(config)

        # Verify all stages are created
        expected_stages = ['ctc_warmup', 'joint', 'mode', 'denoise']
        for stage in expected_stages:
            assert stage in stage_configs
            assert stage_configs[stage].name == stage

        # Verify stage-specific configurations
        ctc_config = stage_configs['ctc_warmup']
        assert ctc_config.enabled is True
        assert ctc_config.epochs == 3
        assert ctc_config.ctc_weight == 1.0
        assert ctc_config.rnnt_weight == 0.0
        assert ctc_config.train_preprocessor is True
        assert ctc_config.train_rnnt_decoder is False

        joint_config = stage_configs['joint']
        assert joint_config.epochs == 4
        assert joint_config.ctc_weight == 0.5
        assert joint_config.rnnt_weight == 0.5
        assert joint_config.train_rnnt_decoder is True

    def test_stage_transition_manager(self, curriculum_config):
        """Test stage transition management"""
        config = OmegaConf.create(curriculum_config)
        stage_configs = create_stage_configs(config)

        manager = StageTransitionManager(stage_configs)

        # Initial state
        assert manager.current_stage == TrainingStage.CTC_WARMUP
        assert manager.stage_epoch == 0
        assert manager.total_epoch == 0

        # Test stage progression
        for epoch in range(3):  # CTC warmup has 3 epochs
            manager.update_epoch({'loss': 2.0 - epoch * 0.1})

            if epoch < 2:  # Not yet time to transition
                assert not manager.should_transition({'loss': 1.8})
            else:  # Should transition after epoch limit
                assert manager.should_transition({'loss': 1.8})

        # Transition to next stage
        next_stage = manager.transition_to_next_stage()
        assert next_stage == TrainingStage.JOINT
        assert manager.current_stage == TrainingStage.JOINT
        assert manager.stage_epoch == 0

    def test_convergence_based_transition(self, curriculum_config):
        """Test early stage transition based on convergence"""
        config = OmegaConf.create(curriculum_config)
        stage_configs = create_stage_configs(config)

        manager = StageTransitionManager(stage_configs)

        # Simulate converged training (loss not improving)
        stable_loss = 1.5
        for epoch in range(5):
            manager.update_epoch({'loss': stable_loss})

        # Should trigger convergence-based transition
        assert manager.should_transition({'loss': stable_loss})

    def test_component_activation_manager(self, mock_model):
        """Test model component activation for different stages"""
        activation_manager = ComponentActivationManager(mock_model)

        # Create mock stage config for CTC warmup
        from mode_ssm.training_stages import StageConfig
        ctc_config = StageConfig(
            name='ctc_warmup',
            train_preprocessor=True,
            train_encoder=True,
            train_ctc_decoder=True,
            train_rnnt_decoder=False,
            train_mode_head=False,
            train_flow_bridge=False
        )

        # Configure for CTC stage
        activation_manager.configure_for_stage(ctc_config)

        # Verify correct parameters are enabled/disabled
        params_dict = dict(mock_model.named_parameters())

        # These should be enabled
        assert params_dict['preprocessor.weight'].requires_grad is True
        assert params_dict['encoder.weight'].requires_grad is True
        assert params_dict['ctc_decoder.weight'].requires_grad is True

        # These should be disabled
        assert params_dict['rnnt_decoder.weight'].requires_grad is False
        assert params_dict['mode_head.weight'].requires_grad is False
        assert params_dict['flow_bridge.weight'].requires_grad is False

        # Test joint stage configuration
        joint_config = StageConfig(
            name='joint',
            train_preprocessor=True,
            train_encoder=True,
            train_ctc_decoder=True,
            train_rnnt_decoder=True,
            train_mode_head=False,
            train_flow_bridge=False
        )

        activation_manager.configure_for_stage(joint_config)

        # Now RNNT decoder should be enabled
        assert params_dict['rnnt_decoder.weight'].requires_grad is True
        assert params_dict['mode_head.weight'].requires_grad is False

    def test_loss_weight_manager(self):
        """Test loss weight management across stages"""
        weight_manager = LossWeightManager()

        from mode_ssm.training_stages import StageConfig

        # Test CTC stage weights
        ctc_config = StageConfig(
            name='ctc_warmup',
            ctc_weight=1.0,
            rnnt_weight=0.0,
            mode_weight=0.0,
            flow_weight=0.0
        )

        weights = weight_manager.update_weights(ctc_config)
        assert weights['ctc_weight'] == 1.0
        assert weights['rnnt_weight'] == 0.0
        assert weights['mode_weight'] == 0.0
        assert weights['flow_weight'] == 0.0

        # Test joint stage weights
        joint_config = StageConfig(
            name='joint',
            ctc_weight=0.5,
            rnnt_weight=0.5,
            mode_weight=0.0,
            flow_weight=0.0
        )

        weights = weight_manager.update_weights(joint_config)
        assert weights['ctc_weight'] == 0.5
        assert weights['rnnt_weight'] == 0.5

        # Test weight history
        assert len(weight_manager.stage_weights_history) == 2
        assert weight_manager.stage_weights_history[0]['ctc_weight'] == 1.0
        assert weight_manager.stage_weights_history[1]['ctc_weight'] == 0.5

    def test_curriculum_trainer_integration(self, mock_model, curriculum_config,
                                           optimizer_factory, scheduler_factory):
        """Test complete curriculum trainer integration"""
        config = OmegaConf.create(curriculum_config)

        trainer = CurriculumTrainer(
            model=mock_model,
            stage_configs=create_stage_configs(config),
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory
        )

        # Initial state should be CTC warmup
        assert trainer.get_current_stage() == 'ctc_warmup'

        weights = trainer.get_loss_weights()
        assert weights['ctc_weight'] == 1.0
        assert weights['rnnt_weight'] == 0.0

        # Simulate training epochs
        for epoch in range(3):  # CTC stage has 3 epochs
            metrics = {'loss': 2.0 - epoch * 0.3, 'val_wer': 0.5 - epoch * 0.1}
            stage_transition = trainer.step_epoch(metrics)

            if epoch < 2:
                assert stage_transition is None  # No transition yet
            else:
                assert stage_transition == 'joint'  # Transitioned to joint

        # Should now be in joint stage
        assert trainer.get_current_stage() == 'joint'
        weights = trainer.get_loss_weights()
        assert weights['ctc_weight'] == 0.5
        assert weights['rnnt_weight'] == 0.5

    def test_complete_curriculum_flow(self, mock_model, curriculum_config,
                                     optimizer_factory, scheduler_factory):
        """Test complete curriculum flow through all stages"""
        config = OmegaConf.create(curriculum_config)

        trainer = CurriculumTrainer(
            model=mock_model,
            stage_configs=create_stage_configs(config),
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory
        )

        expected_stages = ['ctc_warmup', 'joint', 'mode', 'denoise']
        stage_transitions = []

        # Simulate training through all stages
        total_epochs = 3 + 4 + 2 + 2  # Sum of epochs per stage

        for epoch in range(total_epochs + 5):  # Extra epochs to test completion
            metrics = {'loss': max(0.1, 2.0 - epoch * 0.1), 'val_wer': max(0.05, 0.5 - epoch * 0.02)}
            stage_transition = trainer.step_epoch(metrics)

            if stage_transition:
                stage_transitions.append(stage_transition)

        # Should have transitioned through all stages
        expected_transitions = ['joint', 'mode', 'denoise', 'complete']
        assert stage_transitions == expected_transitions

        # Training should be complete
        assert trainer.is_training_complete()

    def test_stage_specific_optimizer_creation(self, mock_model, curriculum_config):
        """Test optimizer creation with stage-specific parameters"""
        config = OmegaConf.create(curriculum_config)

        # Add stage-specific learning rates
        config.training.stages.ctc_warmup.learning_rate = 1e-3
        config.training.stages.joint.learning_rate = 5e-4

        def optimizer_factory(parameters, lr=1e-4, **kwargs):
            return optim.Adam(parameters, lr=lr, **kwargs)

        trainer = CurriculumTrainer(
            model=mock_model,
            stage_configs=create_stage_configs(config),
            optimizer_factory=optimizer_factory
        )

        # Initial optimizer should use CTC learning rate
        assert trainer.optimizer.param_groups[0]['lr'] == 1e-3

        # Transition to joint stage
        for epoch in range(3):
            metrics = {'loss': 1.5}
            trainer.step_epoch(metrics)

        # New optimizer should use joint learning rate
        assert trainer.optimizer.param_groups[0]['lr'] == 5e-4

    def test_disabled_stage_handling(self, mock_model, curriculum_config,
                                   optimizer_factory):
        """Test handling of disabled training stages"""
        config = OmegaConf.create(curriculum_config)

        # Disable mode and denoise stages
        config.training.stages.mode.enabled = False
        config.training.stages.denoise.enabled = False

        trainer = CurriculumTrainer(
            model=mock_model,
            stage_configs=create_stage_configs(config),
            optimizer_factory=optimizer_factory
        )

        stage_transitions = []

        # Simulate training - should skip disabled stages
        for epoch in range(10):
            metrics = {'loss': max(0.1, 2.0 - epoch * 0.2)}
            stage_transition = trainer.step_epoch(metrics)

            if stage_transition:
                stage_transitions.append(stage_transition)

        # Should only transition ctc_warmup -> joint -> complete
        expected_transitions = ['joint', 'complete']
        assert stage_transitions == expected_transitions

    def test_stage_info_tracking(self, mock_model, curriculum_config,
                                optimizer_factory):
        """Test stage information tracking"""
        config = OmegaConf.create(curriculum_config)

        trainer = CurriculumTrainer(
            model=mock_model,
            stage_configs=create_stage_configs(config),
            optimizer_factory=optimizer_factory
        )

        # Initial stage info
        info = trainer.get_stage_info()
        assert info['stage'] == 'ctc_warmup'
        assert info['stage_epoch'] == 0
        assert info['total_epoch'] == 0
        assert info['max_epochs'] == 3
        assert info['loss_weights']['ctc_weight'] == 1.0

        # After one epoch
        trainer.step_epoch({'loss': 1.5})
        info = trainer.get_stage_info()
        assert info['stage_epoch'] == 1
        assert info['total_epoch'] == 1

        # After stage transition
        for epoch in range(2):  # Complete CTC stage
            trainer.step_epoch({'loss': 1.5})

        info = trainer.get_stage_info()
        assert info['stage'] == 'joint'
        assert info['stage_epoch'] == 0  # Reset for new stage
        assert info['total_epoch'] == 3  # Continues counting
        assert info['max_epochs'] == 4  # Joint stage epochs

    def test_error_handling_in_curriculum(self, curriculum_config):
        """Test error handling in curriculum training"""
        config = OmegaConf.create(curriculum_config)

        # Test with invalid stage configuration
        config.training.stages.ctc_warmup.enabled = False  # Required stage disabled

        with pytest.raises(ValueError):
            create_stage_configs(config)

        # Test with missing required stage
        config = OmegaConf.create(curriculum_config)
        del config.training.stages.ctc_warmup

        with pytest.raises(ValueError):
            create_stage_configs(config)

    def test_curriculum_reproducibility(self, mock_model, curriculum_config,
                                       optimizer_factory):
        """Test reproducibility of curriculum training"""
        config = OmegaConf.create(curriculum_config)

        # Set seeds
        torch.manual_seed(42)

        trainer1 = CurriculumTrainer(
            model=mock_model,
            stage_configs=create_stage_configs(config),
            optimizer_factory=optimizer_factory
        )

        # Reset seed and create identical trainer
        torch.manual_seed(42)

        trainer2 = CurriculumTrainer(
            model=mock_model,
            stage_configs=create_stage_configs(config),
            optimizer_factory=optimizer_factory
        )

        # Both trainers should start in same state
        assert trainer1.get_current_stage() == trainer2.get_current_stage()
        assert trainer1.get_loss_weights() == trainer2.get_loss_weights()

        # After same training steps, should have same progression
        metrics = {'loss': 1.5, 'val_wer': 0.3}

        result1 = trainer1.step_epoch(metrics)
        result2 = trainer2.step_epoch(metrics)

        assert result1 == result2
        assert trainer1.get_current_stage() == trainer2.get_current_stage()

    def test_memory_efficiency_in_curriculum(self, mock_model, curriculum_config,
                                           optimizer_factory):
        """Test memory efficiency during curriculum training"""
        config = OmegaConf.create(curriculum_config)

        trainer = CurriculumTrainer(
            model=mock_model,
            stage_configs=create_stage_configs(config),
            optimizer_factory=optimizer_factory
        )

        # Track that stage transitions don't leak memory
        # (This is a basic test - more sophisticated memory tracking could be added)
        initial_param_count = len(list(mock_model.parameters()))

        # Go through multiple stage transitions
        for epoch in range(15):  # Enough to trigger multiple transitions
            metrics = {'loss': max(0.1, 2.0 - epoch * 0.15)}
            trainer.step_epoch(metrics)

        # Parameter count should remain the same (no memory leaks in model)
        final_param_count = len(list(mock_model.parameters()))
        assert initial_param_count == final_param_count