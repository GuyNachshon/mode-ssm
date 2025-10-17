#!/usr/bin/env python3
"""
Submission validation script for Brain-to-Text 2025 competition.
Validates CSV submission files for format compliance and data integrity.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mode_ssm.submission_formatter import validate_submission_format, ensure_chronological_order


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


class SubmissionValidator:
    """Comprehensive submission validation tool"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def validate_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Validate a single submission file.

        Args:
            filepath: Path to CSV submission file

        Returns:
            Validation results dictionary
        """
        results = {
            'file_path': str(filepath),
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'file_info': {}
        }

        self.logger.info(f"Validating submission: {filepath}")

        # Check file existence
        if not filepath.exists():
            results['errors'].append(f"File not found: {filepath}")
            return results

        # Get file info
        file_stat = filepath.stat()
        results['file_info'] = {
            'size_bytes': file_stat.st_size,
            'size_mb': round(file_stat.st_size / (1024 * 1024), 2)
        }

        # Basic format validation
        is_valid, errors = validate_submission_format(filepath)
        results['is_valid'] = is_valid
        results['errors'] = errors

        if not is_valid:
            self.logger.error(f"Validation failed with {len(errors)} errors")
            for error in errors:
                self.logger.error(f"  - {error}")
            return results

        # Load and analyze data
        try:
            df = pd.read_csv(filepath)
            results['statistics'] = self._analyze_submission(df)
            results['warnings'] = self._check_warnings(df)

        except Exception as e:
            results['errors'].append(f"Error analyzing submission: {str(e)}")
            results['is_valid'] = False

        return results

    def _analyze_submission(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze submission statistics"""
        stats = {
            'total_records': len(df),
            'num_sessions': df['session'].nunique(),
            'sessions': list(df['session'].unique()),
            'blocks_per_session': {},
            'trials_per_block': {},
            'prediction_stats': {}
        }

        # Per-session analysis
        for session in stats['sessions']:
            session_df = df[df['session'] == session]
            blocks = sorted(session_df['block'].unique())
            stats['blocks_per_session'][session] = {
                'count': len(blocks),
                'range': f"{min(blocks)}-{max(blocks)}",
                'blocks': blocks
            }

            # Per-block analysis
            for block in blocks:
                block_df = session_df[session_df['block'] == block]
                trials = sorted(block_df['trial'].unique())
                key = f"{session}_block_{block}"
                stats['trials_per_block'][key] = {
                    'count': len(trials),
                    'range': f"{min(trials)}-{max(trials)}",
                    'trials': trials
                }

        # Prediction analysis
        predictions = df['prediction'].astype(str)
        stats['prediction_stats'] = {
            'empty_count': (predictions == '').sum(),
            'avg_length': predictions.str.len().mean(),
            'min_length': predictions.str.len().min(),
            'max_length': predictions.str.len().max(),
            'unique_predictions': predictions.nunique(),
            'most_common': predictions.value_counts().head(5).to_dict()
        }

        return stats

    def _check_warnings(self, df: pd.DataFrame) -> List[str]:
        """Check for potential issues that aren't errors but may indicate problems"""
        warnings = []

        # Check for very short predictions
        short_predictions = df[df['prediction'].str.len() < 3]
        if not short_predictions.empty:
            warnings.append(f"Found {len(short_predictions)} predictions shorter than 3 characters")

        # Check for very long predictions
        long_predictions = df[df['prediction'].str.len() > 100]
        if not long_predictions.empty:
            warnings.append(f"Found {len(long_predictions)} predictions longer than 100 characters")

        # Check for repeated predictions
        prediction_counts = df['prediction'].value_counts()
        repeated = prediction_counts[prediction_counts > 10]
        if not repeated.empty:
            warnings.append(f"Found {len(repeated)} predictions repeated more than 10 times")

        # Check for gaps in block/trial numbering
        for session in df['session'].unique():
            session_df = df[df['session'] == session]

            for block in session_df['block'].unique():
                block_df = session_df[session_df['block'] == block]
                trials = sorted(block_df['trial'].unique())

                # Check for gaps in trial numbering
                expected_trials = list(range(min(trials), max(trials) + 1))
                missing_trials = set(expected_trials) - set(trials)
                if missing_trials:
                    warnings.append(
                        f"Missing trials in {session}, block {block}: {sorted(missing_trials)}"
                    )

        return warnings

    def validate_multiple_files(self, filepaths: List[Path]) -> List[Dict[str, Any]]:
        """Validate multiple submission files"""
        results = []

        for filepath in filepaths:
            result = self.validate_file(filepath)
            results.append(result)

        return results

    def generate_report(self, results: List[Dict[str, Any]], output_path: Optional[Path] = None):
        """Generate validation report"""
        report = {
            'validation_summary': {
                'total_files': len(results),
                'valid_files': sum(1 for r in results if r['is_valid']),
                'invalid_files': sum(1 for r in results if not r['is_valid']),
                'files_with_warnings': sum(1 for r in results if r['warnings'])
            },
            'file_results': results,
            'overall_statistics': self._compute_overall_stats(results)
        }

        # Print summary to console
        self._print_report_summary(report)

        # Save detailed report if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Detailed report saved to: {output_path}")

        return report

    def _compute_overall_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute overall statistics across all files"""
        valid_results = [r for r in results if r['is_valid'] and r['statistics']]

        if not valid_results:
            return {}

        total_records = sum(r['statistics']['total_records'] for r in valid_results)
        total_sessions = sum(r['statistics']['num_sessions'] for r in valid_results)

        return {
            'total_records_across_files': total_records,
            'total_sessions_across_files': total_sessions,
            'avg_records_per_file': total_records / len(valid_results),
            'avg_sessions_per_file': total_sessions / len(valid_results)
        }

    def _print_report_summary(self, report: Dict[str, Any]):
        """Print validation report summary to console"""
        summary = report['validation_summary']

        print("\n" + "="*60)
        print("SUBMISSION VALIDATION REPORT")
        print("="*60)
        print(f"Total files validated: {summary['total_files']}")
        print(f"Valid submissions: {summary['valid_files']}")
        print(f"Invalid submissions: {summary['invalid_files']}")
        print(f"Files with warnings: {summary['files_with_warnings']}")

        if summary['invalid_files'] > 0:
            print(f"\n❌ {summary['invalid_files']} INVALID SUBMISSIONS FOUND")
        else:
            print(f"\n✅ ALL SUBMISSIONS ARE VALID")

        # Print individual file results
        print("\nFile Details:")
        print("-" * 60)

        for result in report['file_results']:
            filepath = Path(result['file_path']).name
            status = "✅ VALID" if result['is_valid'] else "❌ INVALID"
            warnings_count = len(result.get('warnings', []))

            print(f"{filepath}: {status}")

            if result.get('file_info'):
                size_mb = result['file_info']['size_mb']
                print(f"  Size: {size_mb} MB")

            if result.get('statistics'):
                stats = result['statistics']
                print(f"  Records: {stats['total_records']}, Sessions: {stats['num_sessions']}")

            if result['errors']:
                print(f"  Errors ({len(result['errors'])}):")
                for error in result['errors'][:3]:  # Show first 3 errors
                    print(f"    - {error}")
                if len(result['errors']) > 3:
                    print(f"    ... and {len(result['errors']) - 3} more errors")

            if warnings_count > 0:
                print(f"  Warnings: {warnings_count}")
                if self.verbose:
                    for warning in result['warnings']:
                        print(f"    - {warning}")

            print()


def main():
    """Main validation entry point"""
    parser = argparse.ArgumentParser(
        description="Validate Brain-to-Text 2025 competition submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single file
  python validate_submission.py submission.csv

  # Validate multiple files
  python validate_submission.py submission1.csv submission2.csv

  # Validate all CSV files in directory
  python validate_submission.py submissions/*.csv

  # Generate detailed report
  python validate_submission.py submission.csv --report-output validation_report.json

  # Verbose output with warnings
  python validate_submission.py submission.csv --verbose
        """
    )

    parser.add_argument(
        'files',
        nargs='+',
        help='Submission CSV files to validate'
    )

    parser.add_argument(
        '--report-output',
        type=Path,
        help='Path to save detailed validation report (JSON format)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed warnings'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Convert file arguments to Path objects
    filepaths = []
    for file_arg in args.files:
        filepath = Path(file_arg)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            continue
        filepaths.append(filepath)

    if not filepaths:
        logger.error("No valid files to validate")
        sys.exit(1)

    # Create validator and run validation
    validator = SubmissionValidator(verbose=args.verbose)
    results = validator.validate_multiple_files(filepaths)

    # Generate and display report
    report = validator.generate_report(results, args.report_output)

    # Exit with error code if any files are invalid
    invalid_count = report['validation_summary']['invalid_files']
    if invalid_count > 0:
        logger.error(f"Validation failed: {invalid_count} invalid files found")
        sys.exit(1)
    else:
        logger.info("All submissions are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()