#!/usr/bin/env python3
"""
Neural drift analysis script for MODE-SSM.
Analyzes neural signal drift patterns across sessions and time.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.brain2text import Brain2TextDataset, collate_batch
from mode_ssm.models.tta_loop import TTALoop, TTAConfig, SessionStatsAdapter
from mode_ssm.models.mode_ssm_model import MODESSMModel, MODESSMConfig
from mode_ssm.checkpoint_manager import CheckpointManager


logger = logging.getLogger(__name__)


class NeuralDriftAnalyzer:
    """Analyzes neural signal drift patterns and TTA effectiveness"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Analysis results
        self.drift_statistics = {}
        self.session_comparisons = {}
        self.tta_effectiveness = {}

        # Load model if provided
        self.model = None
        if config.model_analysis.enabled:
            self._load_model()

        logger.info(f"Neural drift analyzer initialized on {self.device}")

    def _load_model(self):
        """Load trained model for analysis"""
        model_config = MODESSMConfig(
            d_model=self.config.model.d_model,
            d_state=self.config.model.d_state,
            d_conv=self.config.model.d_conv,
            expand=self.config.model.expand,
            num_channels=self.config.model.num_channels,
            encoder_layers=self.config.model.encoder_layers,
            vocab_size=self.config.model.vocab_size
        )

        self.model = MODESSMModel(model_config).to(self.device)
        self.model.eval()

        # Load checkpoint
        if self.config.checkpoint.path:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=self.config.checkpoint.dir,
                monitor_metric='val_wer'
            )

            checkpoint_path = Path(self.config.checkpoint.path)
            if checkpoint_path.exists():
                checkpoint_manager.restore_model(self.model, checkpoint_path)
                logger.info(f"Model loaded from: {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")

    def analyze_dataset_drift(self, dataset_path: Path) -> Dict[str, Any]:
        """
        Analyze neural drift patterns in dataset.

        Args:
            dataset_path: Path to HDF5 dataset

        Returns:
            Dictionary with drift analysis results
        """
        logger.info(f"Analyzing neural drift in dataset: {dataset_path}")

        # Load dataset
        dataset = Brain2TextDataset(
            hdf5_path=dataset_path,
            cache_data=False,
            filter_quality=self.config.analysis.filter_quality
        )

        # Group samples by session
        session_data = self._group_by_session(dataset)

        # Analyze each session
        session_analyses = {}
        for session_id, samples in session_data.items():
            if len(samples) >= self.config.analysis.min_samples_per_session:
                session_analyses[session_id] = self._analyze_session_drift(
                    samples, session_id
                )

        # Cross-session comparison
        cross_session_analysis = self._compare_sessions(session_analyses)

        # Temporal drift analysis
        temporal_analysis = self._analyze_temporal_drift(session_data)

        return {
            'session_analyses': session_analyses,
            'cross_session_analysis': cross_session_analysis,
            'temporal_analysis': temporal_analysis,
            'dataset_statistics': self._compute_dataset_statistics(dataset)
        }

    def _group_by_session(self, dataset: Brain2TextDataset) -> Dict[str, List[Dict]]:
        """Group dataset samples by session ID"""
        session_data = {}

        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                session_id = sample['session_id']

                if session_id not in session_data:
                    session_data[session_id] = []

                session_data[session_id].append({
                    'index': i,
                    'neural_features': sample['neural_features'],
                    'sequence_length': sample['sequence_length'],
                    'block_num': sample['block_num'],
                    'trial_num': sample['trial_num']
                })

            except Exception as e:
                logger.warning(f"Error loading sample {i}: {e}")
                continue

        logger.info(f"Found {len(session_data)} sessions with samples")
        return session_data

    def _analyze_session_drift(self, samples: List[Dict], session_id: str) -> Dict[str, Any]:
        """Analyze drift patterns within a single session"""
        logger.info(f"Analyzing drift in session: {session_id}")

        # Extract neural features
        features = [s['neural_features'] for s in samples]
        blocks = [s['block_num'] for s in samples]
        trials = [s['trial_num'] for s in samples]

        # Compute statistics over time (ordered by block, trial)
        sorted_indices = sorted(range(len(samples)), key=lambda i: (blocks[i], trials[i]))
        sorted_features = [features[i] for i in sorted_indices]

        # Channel-wise statistics over time
        channel_stats_over_time = []
        window_size = self.config.analysis.drift_window_size

        for i in range(0, len(sorted_features) - window_size + 1, window_size // 2):
            window_features = sorted_features[i:i + window_size]

            if window_features:
                # Stack features and compute statistics
                stacked_features = torch.stack(window_features)  # [window, seq_len, channels]

                # Compute per-channel statistics
                channel_means = stacked_features.mean(dim=(0, 1))  # [channels]
                channel_stds = stacked_features.std(dim=(0, 1))   # [channels]
                channel_medians = stacked_features.median(dim=1)[0].median(dim=0)[0]  # [channels]

                channel_stats_over_time.append({
                    'window_start': i,
                    'window_end': min(i + window_size, len(sorted_features)),
                    'channel_means': channel_means,
                    'channel_stds': channel_stds,
                    'channel_medians': channel_medians,
                    'num_samples': len(window_features)
                })

        # Compute drift metrics
        drift_metrics = self._compute_drift_metrics(channel_stats_over_time)

        # Per-block analysis
        block_analysis = self._analyze_blocks_in_session(samples)

        return {
            'session_id': session_id,
            'num_samples': len(samples),
            'num_blocks': len(set(blocks)),
            'channel_stats_over_time': channel_stats_over_time,
            'drift_metrics': drift_metrics,
            'block_analysis': block_analysis
        }

    def _compute_drift_metrics(self, stats_over_time: List[Dict]) -> Dict[str, float]:
        """Compute quantitative drift metrics"""
        if len(stats_over_time) < 2:
            return {'insufficient_data': True}

        # Extract channel means over time
        means_over_time = [s['channel_means'] for s in stats_over_time]
        stds_over_time = [s['channel_stds'] for s in stats_over_time]

        # Convert to tensor for easier computation
        means_tensor = torch.stack(means_over_time)  # [time_windows, channels]
        stds_tensor = torch.stack(stds_over_time)    # [time_windows, channels]

        # Compute drift metrics
        metrics = {}

        # Mean drift: change in channel means over time
        first_means = means_tensor[0]
        last_means = means_tensor[-1]
        mean_drift = torch.norm(last_means - first_means).item()
        metrics['mean_drift_magnitude'] = mean_drift

        # Variance drift: change in channel variances over time
        first_stds = stds_tensor[0]
        last_stds = stds_tensor[-1]
        std_drift = torch.norm(last_stds - first_stds).item()
        metrics['std_drift_magnitude'] = std_drift

        # Temporal correlation: how correlated are channel statistics over time
        mean_correlations = []
        for ch in range(means_tensor.shape[1]):
            channel_means_over_time = means_tensor[:, ch]
            if len(channel_means_over_time) > 2:
                # Compute temporal trend
                time_indices = torch.arange(len(channel_means_over_time), dtype=torch.float32)
                correlation = torch.corrcoef(torch.stack([time_indices, channel_means_over_time]))[0, 1]
                if not torch.isnan(correlation):
                    mean_correlations.append(correlation.item())

        if mean_correlations:
            metrics['temporal_correlation_mean'] = np.mean(mean_correlations)
            metrics['temporal_correlation_std'] = np.std(mean_correlations)

        # Drift rate: average change per window
        if len(means_over_time) > 1:
            window_to_window_changes = []
            for i in range(1, len(means_over_time)):
                change = torch.norm(means_tensor[i] - means_tensor[i-1]).item()
                window_to_window_changes.append(change)
            metrics['average_drift_rate'] = np.mean(window_to_window_changes)
            metrics['max_drift_rate'] = np.max(window_to_window_changes)

        # Channel-wise drift analysis
        per_channel_drift = torch.norm(means_tensor[-1] - means_tensor[0], dim=0)  # [channels]
        metrics['max_channel_drift'] = per_channel_drift.max().item()
        metrics['mean_channel_drift'] = per_channel_drift.mean().item()
        metrics['channels_with_high_drift'] = (per_channel_drift > per_channel_drift.mean() + per_channel_drift.std()).sum().item()

        return metrics

    def _analyze_blocks_in_session(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze drift patterns across blocks within a session"""
        # Group samples by block
        block_data = {}
        for sample in samples:
            block_num = sample['block_num']
            if block_num not in block_data:
                block_data[block_num] = []
            block_data[block_num].append(sample['neural_features'])

        # Compute per-block statistics
        block_stats = {}
        for block_num, features_list in block_data.items():
            if len(features_list) >= 3:  # Need minimum samples
                stacked_features = torch.stack(features_list)
                block_stats[block_num] = {
                    'mean': stacked_features.mean(dim=(0, 1)),  # [channels]
                    'std': stacked_features.std(dim=(0, 1)),   # [channels]
                    'num_samples': len(features_list)
                }

        # Cross-block comparisons
        block_comparisons = {}
        block_numbers = sorted(block_stats.keys())

        for i, block1 in enumerate(block_numbers):
            for j, block2 in enumerate(block_numbers[i+1:], i+1):
                stats1 = block_stats[block1]
                stats2 = block_stats[block2]

                # Compare means
                mean_diff = torch.norm(stats1['mean'] - stats2['mean']).item()

                # Compare stds
                std_diff = torch.norm(stats1['std'] - stats2['std']).item()

                block_comparisons[f"block_{block1}_vs_{block2}"] = {
                    'mean_difference': mean_diff,
                    'std_difference': std_diff,
                    'block_distance': abs(block2 - block1)
                }

        return {
            'block_stats': block_stats,
            'block_comparisons': block_comparisons,
            'num_blocks': len(block_stats)
        }

    def _compare_sessions(self, session_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Compare drift patterns across different sessions"""
        if len(session_analyses) < 2:
            return {'insufficient_sessions': True}

        session_ids = list(session_analyses.keys())
        comparisons = {}

        # Pairwise session comparisons
        for i, session1 in enumerate(session_ids):
            for j, session2 in enumerate(session_ids[i+1:], i+1):
                analysis1 = session_analyses[session1]
                analysis2 = session_analyses[session2]

                # Compare drift metrics
                drift1 = analysis1['drift_metrics']
                drift2 = analysis2['drift_metrics']

                comparison = {}
                for metric in ['mean_drift_magnitude', 'std_drift_magnitude', 'average_drift_rate']:
                    if metric in drift1 and metric in drift2:
                        comparison[f'{metric}_diff'] = abs(drift1[metric] - drift2[metric])
                        comparison[f'{metric}_ratio'] = drift1[metric] / max(drift2[metric], 1e-6)

                # Compare session characteristics
                comparison['sample_count_ratio'] = analysis1['num_samples'] / max(analysis2['num_samples'], 1)
                comparison['block_count_diff'] = abs(analysis1['num_blocks'] - analysis2['num_blocks'])

                comparisons[f"{session1}_vs_{session2}"] = comparison

        # Global session statistics
        global_stats = {
            'num_sessions': len(session_analyses),
            'total_samples': sum(a['num_samples'] for a in session_analyses.values()),
            'mean_drift_range': [
                min(a['drift_metrics'].get('mean_drift_magnitude', 0) for a in session_analyses.values()),
                max(a['drift_metrics'].get('mean_drift_magnitude', 0) for a in session_analyses.values())
            ]
        }

        return {
            'pairwise_comparisons': comparisons,
            'global_statistics': global_stats
        }

    def _analyze_temporal_drift(self, session_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze drift patterns over time across all sessions"""
        # Collect all samples with timestamps (approximated by block/trial)
        all_samples_with_time = []

        for session_id, samples in session_data.items():
            for sample in samples:
                timestamp = sample['block_num'] * 1000 + sample['trial_num']  # Approximate ordering
                all_samples_with_time.append({
                    'timestamp': timestamp,
                    'session_id': session_id,
                    'neural_features': sample['neural_features']
                })

        # Sort by timestamp
        all_samples_with_time.sort(key=lambda x: x['timestamp'])

        # Compute global statistics over time
        window_size = self.config.analysis.global_drift_window_size
        global_stats_over_time = []

        for i in range(0, len(all_samples_with_time) - window_size + 1, window_size // 2):
            window_samples = all_samples_with_time[i:i + window_size]

            if window_samples:
                features = [s['neural_features'] for s in window_samples]
                sessions_in_window = set(s['session_id'] for s in window_samples)

                if len(features) > 1:
                    stacked_features = torch.stack(features)

                    global_stats_over_time.append({
                        'window_start': i,
                        'timestamp_range': [window_samples[0]['timestamp'], window_samples[-1]['timestamp']],
                        'global_mean': stacked_features.mean(dim=(0, 1)),
                        'global_std': stacked_features.std(dim=(0, 1)),
                        'num_sessions': len(sessions_in_window),
                        'num_samples': len(features)
                    })

        # Compute global drift metrics
        global_drift_metrics = {}
        if len(global_stats_over_time) >= 2:
            first_stats = global_stats_over_time[0]
            last_stats = global_stats_over_time[-1]

            global_mean_drift = torch.norm(last_stats['global_mean'] - first_stats['global_mean']).item()
            global_std_drift = torch.norm(last_stats['global_std'] - first_stats['global_std']).item()

            global_drift_metrics = {
                'global_mean_drift': global_mean_drift,
                'global_std_drift': global_std_drift,
                'analysis_time_span': last_stats['timestamp_range'][1] - first_stats['timestamp_range'][0]
            }

        return {
            'global_stats_over_time': global_stats_over_time,
            'global_drift_metrics': global_drift_metrics
        }

    def analyze_tta_effectiveness(self, dataset_path: Path, tta_config: TTAConfig) -> Dict[str, Any]:
        """Analyze effectiveness of test-time adaptation"""
        if self.model is None:
            logger.warning("Model not loaded - skipping TTA effectiveness analysis")
            return {'model_not_loaded': True}

        logger.info("Analyzing TTA effectiveness")

        # Load dataset
        dataset = Brain2TextDataset(
            hdf5_path=dataset_path,
            cache_data=False,
            filter_quality=False
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.tta_analysis.batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )

        # Create TTA loop
        tta_loop = TTALoop(tta_config, self.model)

        # Compare performance with and without TTA
        results_without_tta = []
        results_with_tta = []
        tta_statistics = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= self.config.tta_analysis.max_batches:
                    break

                # Move to device
                batch['neural_features'] = batch['neural_features'].to(self.device)
                batch['sequence_lengths'] = batch['sequence_lengths'].to(self.device)

                # Baseline (without TTA)
                baseline_outputs = self.model.inference(
                    neural_features=batch['neural_features'],
                    sequence_lengths=batch['sequence_lengths'],
                    decode_mode='greedy'
                )

                # With TTA
                adapted_batch = tta_loop.adapt_multi_step(
                    batch, num_steps=tta_config.adaptation_steps
                )

                tta_outputs = self.model.inference(
                    neural_features=adapted_batch['neural_features'],
                    sequence_lengths=adapted_batch['sequence_lengths'],
                    decode_mode='greedy'
                )

                # Store results
                results_without_tta.append(baseline_outputs)
                results_with_tta.append(tta_outputs)
                tta_statistics.append(adapted_batch['adaptation_stats'])

        # Analyze differences
        tta_effectiveness = self._compute_tta_effectiveness(
            results_without_tta, results_with_tta, tta_statistics
        )

        return tta_effectiveness

    def _compute_tta_effectiveness(
        self,
        baseline_results: List[Dict],
        tta_results: List[Dict],
        tta_stats: List[Dict]
    ) -> Dict[str, Any]:
        """Compute TTA effectiveness metrics"""

        # Count prediction changes
        total_predictions = 0
        changed_predictions = 0

        for baseline, tta in zip(baseline_results, tta_results):
            baseline_seqs = baseline['decoded_sequences']
            tta_seqs = tta['decoded_sequences']

            for b_seq, t_seq in zip(baseline_seqs, tta_seqs):
                total_predictions += 1
                if not torch.equal(b_seq, t_seq):
                    changed_predictions += 1

        change_rate = changed_predictions / max(total_predictions, 1)

        # Analyze adaptation statistics
        adaptation_effectiveness = {}
        if tta_stats:
            # Session adaptation metrics
            all_session_adaptations = {}
            feature_changes = []
            entropy_improvements = []

            for stats in tta_stats:
                # Track session adaptations
                for session_id, count in stats.get('session_adaptations', {}).items():
                    if session_id not in all_session_adaptations:
                        all_session_adaptations[session_id] = []
                    all_session_adaptations[session_id].append(count)

                # Track feature changes
                if 'feature_change_magnitude' in stats:
                    feature_changes.append(stats['feature_change_magnitude'])

                # Track entropy changes
                entropy_stats = stats.get('entropy_stats', {})
                if 'mean_entropy' in entropy_stats:
                    entropy_improvements.append(entropy_stats['mean_entropy'])

            adaptation_effectiveness = {
                'sessions_adapted': len(all_session_adaptations),
                'avg_feature_change': np.mean(feature_changes) if feature_changes else 0.0,
                'entropy_trend': {
                    'mean': np.mean(entropy_improvements) if entropy_improvements else 0.0,
                    'std': np.std(entropy_improvements) if len(entropy_improvements) > 1 else 0.0
                }
            }

        return {
            'prediction_change_rate': change_rate,
            'total_predictions_analyzed': total_predictions,
            'changed_predictions': changed_predictions,
            'adaptation_effectiveness': adaptation_effectiveness,
            'tta_statistics_summary': self._summarize_tta_statistics(tta_stats)
        }

    def _summarize_tta_statistics(self, tta_stats: List[Dict]) -> Dict[str, Any]:
        """Summarize TTA statistics across all batches"""
        if not tta_stats:
            return {}

        # Aggregate statistics
        all_feature_changes = []
        all_entropy_stats = []
        session_adaptation_counts = defaultdict(list)

        for stats in tta_stats:
            if 'feature_change_magnitude' in stats:
                all_feature_changes.append(stats['feature_change_magnitude'])

            if 'entropy_stats' in stats and 'mean_entropy' in stats['entropy_stats']:
                all_entropy_stats.append(stats['entropy_stats']['mean_entropy'])

            for session_id, count in stats.get('session_adaptations', {}).items():
                session_adaptation_counts[session_id].append(count)

        summary = {
            'feature_change_statistics': {
                'mean': np.mean(all_feature_changes) if all_feature_changes else 0.0,
                'std': np.std(all_feature_changes) if len(all_feature_changes) > 1 else 0.0,
                'min': np.min(all_feature_changes) if all_feature_changes else 0.0,
                'max': np.max(all_feature_changes) if all_feature_changes else 0.0
            },
            'entropy_statistics': {
                'mean': np.mean(all_entropy_stats) if all_entropy_stats else 0.0,
                'std': np.std(all_entropy_stats) if len(all_entropy_stats) > 1 else 0.0
            },
            'session_adaptation_summary': {
                'num_sessions': len(session_adaptation_counts),
                'total_adaptations': sum(sum(counts) for counts in session_adaptation_counts.values())
            }
        }

        return summary

    def _compute_dataset_statistics(self, dataset: Brain2TextDataset) -> Dict[str, Any]:
        """Compute basic dataset statistics"""
        try:
            stats = dataset.get_statistics()
            return stats
        except Exception as e:
            logger.warning(f"Could not get dataset statistics: {e}")
            return {
                'total_samples': len(dataset),
                'error': str(e)
            }

    def generate_report(self, analysis_results: Dict[str, Any], output_dir: Path):
        """Generate comprehensive analysis report"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw results
        results_file = output_dir / "neural_drift_analysis.json"
        with open(results_file, 'w') as f:
            # Convert tensors to lists for JSON serialization
            serializable_results = self._make_json_serializable(analysis_results)
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Analysis results saved to: {results_file}")

        # Generate summary report
        self._generate_summary_report(analysis_results, output_dir)

        # Generate visualizations if requested
        if self.config.analysis.generate_plots:
            self._generate_visualizations(analysis_results, output_dir)

    def _make_json_serializable(self, obj):
        """Convert tensors and other non-serializable objects for JSON"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _generate_summary_report(self, analysis_results: Dict[str, Any], output_dir: Path):
        """Generate human-readable summary report"""
        report_file = output_dir / "drift_analysis_summary.txt"

        with open(report_file, 'w') as f:
            f.write("NEURAL DRIFT ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            # Dataset overview
            dataset_stats = analysis_results.get('dataset_statistics', {})
            f.write(f"Dataset Statistics:\n")
            f.write(f"  Total samples: {dataset_stats.get('total_samples', 'Unknown')}\n")
            if 'sessions' in dataset_stats:
                f.write(f"  Number of sessions: {len(dataset_stats['sessions'])}\n")
            f.write("\n")

            # Session-level analysis
            session_analyses = analysis_results.get('session_analyses', {})
            f.write(f"Session Analysis ({len(session_analyses)} sessions):\n")

            for session_id, analysis in session_analyses.items():
                f.write(f"\n  Session: {session_id}\n")
                f.write(f"    Samples: {analysis['num_samples']}\n")
                f.write(f"    Blocks: {analysis['num_blocks']}\n")

                drift_metrics = analysis.get('drift_metrics', {})
                if 'mean_drift_magnitude' in drift_metrics:
                    f.write(f"    Mean drift magnitude: {drift_metrics['mean_drift_magnitude']:.4f}\n")
                if 'average_drift_rate' in drift_metrics:
                    f.write(f"    Average drift rate: {drift_metrics['average_drift_rate']:.4f}\n")

            # Cross-session comparison
            cross_session = analysis_results.get('cross_session_analysis', {})
            if 'global_statistics' in cross_session:
                global_stats = cross_session['global_statistics']
                f.write(f"\nCross-Session Analysis:\n")
                f.write(f"  Total sessions analyzed: {global_stats['num_sessions']}\n")
                f.write(f"  Total samples: {global_stats['total_samples']}\n")

            # TTA effectiveness
            if 'tta_effectiveness' in analysis_results:
                tta_results = analysis_results['tta_effectiveness']
                f.write(f"\nTest-Time Adaptation Effectiveness:\n")
                f.write(f"  Prediction change rate: {tta_results.get('prediction_change_rate', 0):.2%}\n")
                f.write(f"  Total predictions analyzed: {tta_results.get('total_predictions_analyzed', 0)}\n")

                adaptation_eff = tta_results.get('adaptation_effectiveness', {})
                f.write(f"  Sessions adapted: {adaptation_eff.get('sessions_adapted', 0)}\n")
                f.write(f"  Average feature change: {adaptation_eff.get('avg_feature_change', 0):.4f}\n")

        logger.info(f"Summary report saved to: {report_file}")

    def _generate_visualizations(self, analysis_results: Dict[str, Any], output_dir: Path):
        """Generate visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use('default')
            sns.set_palette("husl")

            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Plot 1: Session drift magnitudes
            session_analyses = analysis_results.get('session_analyses', {})
            if session_analyses:
                self._plot_session_drift_magnitudes(session_analyses, plots_dir)

            # Plot 2: Temporal drift patterns
            temporal_analysis = analysis_results.get('temporal_analysis', {})
            if 'global_stats_over_time' in temporal_analysis:
                self._plot_temporal_drift_patterns(temporal_analysis, plots_dir)

            # Plot 3: TTA effectiveness
            if 'tta_effectiveness' in analysis_results:
                self._plot_tta_effectiveness(analysis_results['tta_effectiveness'], plots_dir)

            logger.info(f"Visualization plots saved to: {plots_dir}")

        except ImportError:
            logger.warning("Matplotlib/seaborn not available - skipping visualizations")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

    def _plot_session_drift_magnitudes(self, session_analyses: Dict, plots_dir: Path):
        """Plot session drift magnitudes"""
        session_ids = []
        drift_magnitudes = []
        sample_counts = []

        for session_id, analysis in session_analyses.items():
            drift_metrics = analysis.get('drift_metrics', {})
            if 'mean_drift_magnitude' in drift_metrics:
                session_ids.append(session_id)
                drift_magnitudes.append(drift_metrics['mean_drift_magnitude'])
                sample_counts.append(analysis['num_samples'])

        if session_ids:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Drift magnitudes
            ax1.bar(range(len(session_ids)), drift_magnitudes)
            ax1.set_xlabel('Session')
            ax1.set_ylabel('Mean Drift Magnitude')
            ax1.set_title('Neural Drift Magnitude by Session')
            ax1.set_xticks(range(len(session_ids)))
            ax1.set_xticklabels(session_ids, rotation=45)

            # Sample counts
            ax2.bar(range(len(session_ids)), sample_counts, alpha=0.7)
            ax2.set_xlabel('Session')
            ax2.set_ylabel('Number of Samples')
            ax2.set_title('Sample Count by Session')
            ax2.set_xticks(range(len(session_ids)))
            ax2.set_xticklabels(session_ids, rotation=45)

            plt.tight_layout()
            plt.savefig(plots_dir / "session_drift_magnitudes.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_temporal_drift_patterns(self, temporal_analysis: Dict, plots_dir: Path):
        """Plot temporal drift patterns"""
        global_stats = temporal_analysis.get('global_stats_over_time', [])

        if len(global_stats) > 1:
            timestamps = [s['timestamp_range'][0] for s in global_stats]
            mean_norms = [torch.norm(s['global_mean']).item() for s in global_stats]
            std_norms = [torch.norm(s['global_std']).item() for s in global_stats]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Mean evolution
            ax1.plot(timestamps, mean_norms, marker='o')
            ax1.set_xlabel('Time (approximate)')
            ax1.set_ylabel('Global Mean Norm')
            ax1.set_title('Evolution of Global Mean Over Time')
            ax1.grid(True, alpha=0.3)

            # Std evolution
            ax2.plot(timestamps, std_norms, marker='s', color='orange')
            ax2.set_xlabel('Time (approximate)')
            ax2.set_ylabel('Global Std Norm')
            ax2.set_title('Evolution of Global Standard Deviation Over Time')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(plots_dir / "temporal_drift_patterns.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_tta_effectiveness(self, tta_effectiveness: Dict, plots_dir: Path):
        """Plot TTA effectiveness metrics"""
        # TTA summary metrics
        adaptation_eff = tta_effectiveness.get('adaptation_effectiveness', {})
        change_rate = tta_effectiveness.get('prediction_change_rate', 0)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Prediction change rate
        ax1.bar(['Baseline', 'TTA'], [1.0 - change_rate, change_rate], color=['lightblue', 'orange'])
        ax1.set_ylabel('Prediction Fraction')
        ax1.set_title('TTA Prediction Change Rate')
        ax1.set_ylim(0, 1)

        # Sessions adapted
        sessions_adapted = adaptation_eff.get('sessions_adapted', 0)
        ax2.bar(['Sessions Adapted'], [sessions_adapted], color='green')
        ax2.set_ylabel('Count')
        ax2.set_title('Sessions with TTA Applied')

        # Feature change magnitude
        avg_feature_change = adaptation_eff.get('avg_feature_change', 0)
        ax3.bar(['Average Feature Change'], [avg_feature_change], color='purple')
        ax3.set_ylabel('Magnitude')
        ax3.set_title('Average Feature Adaptation Magnitude')

        # Entropy trend
        entropy_stats = adaptation_eff.get('entropy_trend', {})
        entropy_mean = entropy_stats.get('mean', 0)
        entropy_std = entropy_stats.get('std', 0)
        ax4.bar(['Mean Entropy'], [entropy_mean], yerr=[entropy_std], color='red', capsize=5)
        ax4.set_ylabel('Entropy')
        ax4.set_title('Prediction Entropy with TTA')

        plt.tight_layout()
        plt.savefig(plots_dir / "tta_effectiveness.png", dpi=300, bbox_inches='tight')
        plt.close()


@hydra.main(config_path="../configs", config_name="analyze_drift", version_base="1.3")
def main(config: DictConfig):
    """Main analysis entry point"""
    analyzer = NeuralDriftAnalyzer(config)

    # Analyze dataset drift
    dataset_path = Path(config.data.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    analysis_results = analyzer.analyze_dataset_drift(dataset_path)

    # Analyze TTA effectiveness if model is available
    if config.tta_analysis.enabled and analyzer.model is not None:
        tta_config = TTAConfig(**config.tta_config)
        tta_effectiveness = analyzer.analyze_tta_effectiveness(dataset_path, tta_config)
        analysis_results['tta_effectiveness'] = tta_effectiveness

    # Generate report
    output_dir = Path(config.output.dir)
    analyzer.generate_report(analysis_results, output_dir)

    logger.info("Neural drift analysis completed successfully!")


if __name__ == "__main__":
    main()