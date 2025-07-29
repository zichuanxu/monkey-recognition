"""Data validation utilities."""

import os
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import json

from ..utils.image_utils import validate_image, get_image_files, get_image_info
from ..utils.file_utils import get_directory_structure
from ..utils.logging import LoggerMixin


class DatasetValidator(LoggerMixin):
    """Validator for monkey dataset structure and content."""

    def __init__(self, min_images_per_monkey: int = 5, max_images_per_monkey: int = 10000):
        """Initialize dataset validator.

        Args:
            min_images_per_monkey: Minimum number of images required per monkey.
            max_images_per_monkey: Maximum number of images allowed per monkey.
        """
        self.min_images_per_monkey = min_images_per_monkey
        self.max_images_per_monkey = max_images_per_monkey
        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    def validate_dataset_structure(self, data_dir: str) -> Dict[str, any]:
        """Validate dataset directory structure.

        Args:
            data_dir: Path to dataset directory.

        Returns:
            Validation results dictionary.
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'monkey_info': {}
        }

        # Check if directory exists
        if not os.path.exists(data_dir):
            results['valid'] = False
            results['errors'].append(f"Dataset directory not found: {data_dir}")
            return results

        if not os.path.isdir(data_dir):
            results['valid'] = False
            results['errors'].append(f"Path is not a directory: {data_dir}")
            return results

        # Get subdirectories (monkey IDs)
        try:
            subdirs = [d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))]
        except PermissionError:
            results['valid'] = False
            results['errors'].append(f"Permission denied accessing directory: {data_dir}")
            return results

        if len(subdirs) == 0:
            results['valid'] = False
            results['errors'].append(f"No subdirectories found in {data_dir}")
            return results

        # Validate each monkey directory
        total_images = 0
        valid_monkeys = 0

        for monkey_id in subdirs:
            monkey_dir = os.path.join(data_dir, monkey_id)
            monkey_validation = self._validate_monkey_directory(monkey_dir, monkey_id)

            results['monkey_info'][monkey_id] = monkey_validation

            if monkey_validation['valid']:
                valid_monkeys += 1
                total_images += monkey_validation['image_count']
            else:
                results['errors'].extend([
                    f"Monkey {monkey_id}: {error}"
                    for error in monkey_validation['errors']
                ])
                results['warnings'].extend([
                    f"Monkey {monkey_id}: {warning}"
                    for warning in monkey_validation['warnings']
                ])

        # Overall statistics
        results['statistics'] = {
            'total_monkeys': len(subdirs),
            'valid_monkeys': valid_monkeys,
            'total_images': total_images,
            'average_images_per_monkey': total_images / valid_monkeys if valid_monkeys > 0 else 0
        }

        # Check if we have enough valid monkeys
        if valid_monkeys < 2:
            results['valid'] = False
            results['errors'].append(f"Need at least 2 valid monkeys, found {valid_monkeys}")

        # Final validation status
        if len(results['errors']) > 0:
            results['valid'] = False

        self.logger.info(f"Dataset validation completed. Valid: {results['valid']}")
        self.logger.info(f"Statistics: {results['statistics']}")

        return results

    def _validate_monkey_directory(self, monkey_dir: str, monkey_id: str) -> Dict[str, any]:
        """Validate individual monkey directory.

        Args:
            monkey_dir: Path to monkey directory.
            monkey_id: Monkey identifier.

        Returns:
            Validation results for this monkey.
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'image_count': 0,
            'valid_images': 0,
            'invalid_images': [],
            'image_info': []
        }

        # Get all image files
        image_files = get_image_files(monkey_dir, self.valid_extensions)
        results['image_count'] = len(image_files)

        # Check minimum images requirement
        if len(image_files) < self.min_images_per_monkey:
            results['valid'] = False
            results['errors'].append(
                f"Too few images ({len(image_files)}), minimum required: {self.min_images_per_monkey}"
            )

        # Check maximum images limit
        if len(image_files) > self.max_images_per_monkey:
            results['warnings'].append(
                f"Many images ({len(image_files)}), consider reducing for efficiency"
            )

        # Validate each image
        for image_path in image_files:
            if validate_image(image_path):
                results['valid_images'] += 1

                # Get image info
                info = get_image_info(image_path)
                if info:
                    results['image_info'].append(info)
            else:
                results['invalid_images'].append(image_path)
                results['warnings'].append(f"Invalid image: {os.path.basename(image_path)}")

        # Check if we have enough valid images
        if results['valid_images'] < self.min_images_per_monkey:
            results['valid'] = False
            results['errors'].append(
                f"Too few valid images ({results['valid_images']}), minimum required: {self.min_images_per_monkey}"
            )

        return results

    def validate_train_test_consistency(
        self,
        train_dir: str,
        test_dir: str
    ) -> Dict[str, any]:
        """Validate consistency between training and test datasets.

        Args:
            train_dir: Training dataset directory.
            test_dir: Test dataset directory.

        Returns:
            Validation results dictionary.
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'train_monkeys': set(),
            'test_monkeys': set(),
            'common_monkeys': set(),
            'train_only': set(),
            'test_only': set()
        }

        # Validate individual datasets
        train_validation = self.validate_dataset_structure(train_dir)
        test_validation = self.validate_dataset_structure(test_dir)

        if not train_validation['valid']:
            results['valid'] = False
            results['errors'].append("Training dataset validation failed")
            results['errors'].extend(train_validation['errors'])

        if not test_validation['valid']:
            results['valid'] = False
            results['errors'].append("Test dataset validation failed")
            results['errors'].extend(test_validation['errors'])

        if not results['valid']:
            return results

        # Get monkey sets
        results['train_monkeys'] = set(train_validation['monkey_info'].keys())
        results['test_monkeys'] = set(test_validation['monkey_info'].keys())

        # Find overlaps and differences
        results['common_monkeys'] = results['train_monkeys'] & results['test_monkeys']
        results['train_only'] = results['train_monkeys'] - results['test_monkeys']
        results['test_only'] = results['test_monkeys'] - results['train_monkeys']

        # Check for consistency issues
        if len(results['test_only']) > 0:
            results['warnings'].append(
                f"Test set contains monkeys not in training set: {results['test_only']}"
            )

        if len(results['common_monkeys']) == 0:
            results['valid'] = False
            results['errors'].append("No common monkeys between training and test sets")

        # Log statistics
        self.logger.info(f"Train monkeys: {len(results['train_monkeys'])}")
        self.logger.info(f"Test monkeys: {len(results['test_monkeys'])}")
        self.logger.info(f"Common monkeys: {len(results['common_monkeys'])}")

        return results

    def generate_dataset_report(
        self,
        data_dir: str,
        output_path: Optional[str] = None
    ) -> Dict[str, any]:
        """Generate comprehensive dataset report.

        Args:
            data_dir: Dataset directory path.
            output_path: Optional path to save report as JSON.

        Returns:
            Dataset report dictionary.
        """
        self.logger.info(f"Generating dataset report for {data_dir}")

        # Validate dataset
        validation_results = self.validate_dataset_structure(data_dir)

        # Generate detailed statistics
        report = {
            'dataset_path': data_dir,
            'validation': validation_results,
            'detailed_statistics': self._generate_detailed_statistics(validation_results),
            'recommendations': self._generate_recommendations(validation_results)
        }

        # Save report if path provided
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
                self.logger.info(f"Dataset report saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save report: {e}")

        return report

    def _generate_detailed_statistics(self, validation_results: Dict) -> Dict[str, any]:
        """Generate detailed statistics from validation results.

        Args:
            validation_results: Results from dataset validation.

        Returns:
            Detailed statistics dictionary.
        """
        stats = {
            'image_size_distribution': {},
            'file_format_distribution': {},
            'monkey_image_counts': {},
            'total_file_size': 0
        }

        # Analyze each monkey's data
        for monkey_id, monkey_info in validation_results.get('monkey_info', {}).items():
            stats['monkey_image_counts'][monkey_id] = monkey_info['valid_images']

            # Analyze image properties
            for img_info in monkey_info.get('image_info', []):
                # Image size distribution
                size_key = f"{img_info['width']}x{img_info['height']}"
                stats['image_size_distribution'][size_key] = \
                    stats['image_size_distribution'].get(size_key, 0) + 1

                # File format distribution
                format_key = img_info['format'].lower()
                stats['file_format_distribution'][format_key] = \
                    stats['file_format_distribution'].get(format_key, 0) + 1

                # Total file size
                stats['total_file_size'] += img_info['file_size']

        # Convert file size to MB
        stats['total_file_size_mb'] = stats['total_file_size'] / (1024 * 1024)

        return stats

    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results.

        Args:
            validation_results: Results from dataset validation.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        if not validation_results['valid']:
            recommendations.append("Fix validation errors before proceeding with training")

        stats = validation_results.get('statistics', {})

        # Check dataset size
        if stats.get('total_monkeys', 0) < 10:
            recommendations.append(
                "Consider collecting more monkey classes for better model generalization"
            )

        # Check image distribution
        avg_images = stats.get('average_images_per_monkey', 0)
        if avg_images < 20:
            recommendations.append(
                "Consider collecting more images per monkey for better recognition accuracy"
            )
        elif avg_images > 1000:
            recommendations.append(
                "Consider reducing images per monkey to improve training efficiency"
            )

        # Check for imbalanced dataset
        monkey_counts = []
        for monkey_info in validation_results.get('monkey_info', {}).values():
            monkey_counts.append(monkey_info['valid_images'])

        if monkey_counts:
            min_count = min(monkey_counts)
            max_count = max(monkey_counts)

            if max_count > min_count * 3:
                recommendations.append(
                    "Dataset is imbalanced. Consider balancing the number of images per monkey"
                )

        return recommendations