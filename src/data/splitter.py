"""Dataset splitting utilities."""

import os
import shutil
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

from ..utils.image_utils import get_image_files
from ..utils.file_utils import ensure_dir
from ..utils.logging import LoggerMixin


class DatasetSplitter(LoggerMixin):
    """Utility for splitting datasets into train/validation/test sets."""

    def __init__(self, random_state: int = 42):
        """Initialize dataset splitter.

        Args:
            random_state: Random state for reproducible splits.
        """
        self.random_state = random_state

    def split_monkey_dataset(
        self,
        source_dir: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_images_per_split: int = 2,
        copy_files: bool = True
    ) -> Dict[str, str]:
        """Split monkey dataset into train/val/test sets.

        Args:
            source_dir: Source directory with monkey subdirectories.
            output_dir: Output directory for split datasets.
            train_ratio: Training set ratio.
            val_ratio: Validation set ratio.
            test_ratio: Test set ratio.
            min_images_per_split: Minimum images required per split for each monkey.
            copy_files: Whether to copy files or just create file lists.

        Returns:
            Dictionary with paths to split datasets.
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        self.logger.info(f"Splitting dataset from {source_dir}")
        self.logger.info(f"Ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")

        # Create output directories
        split_dirs = {
            'train': os.path.join(output_dir, 'train'),
            'val': os.path.join(output_dir, 'val'),
            'test': os.path.join(output_dir, 'test')
        }

        for split_dir in split_dirs.values():
            ensure_dir(split_dir)

        # Get all monkey directories
        monkey_dirs = [d for d in os.listdir(source_dir)
                      if os.path.isdir(os.path.join(source_dir, d))]

        split_stats = {
            'total_monkeys': len(monkey_dirs),
            'valid_monkeys': 0,
            'skipped_monkeys': [],
            'split_counts': {'train': 0, 'val': 0, 'test': 0}
        }

        # Process each monkey
        for monkey_id in monkey_dirs:
            monkey_dir = os.path.join(source_dir, monkey_id)
            images = get_image_files(monkey_dir)

            if len(images) < min_images_per_split * 3:
                self.logger.warning(
                    f"Monkey {monkey_id} has only {len(images)} images, "
                    f"minimum required: {min_images_per_split * 3}. Skipping."
                )
                split_stats['skipped_monkeys'].append(monkey_id)
                continue

            # Split images for this monkey
            monkey_splits = self._split_monkey_images(
                images, train_ratio, val_ratio, test_ratio, min_images_per_split
            )

            # Copy or link files to split directories
            for split_name, split_images in monkey_splits.items():
                split_monkey_dir = os.path.join(split_dirs[split_name], monkey_id)
                ensure_dir(split_monkey_dir)

                for img_path in split_images:
                    img_name = os.path.basename(img_path)
                    dst_path = os.path.join(split_monkey_dir, img_name)

                    if copy_files:
                        shutil.copy2(img_path, dst_path)
                    else:
                        # Create symbolic link (Unix-like systems)
                        try:
                            os.symlink(img_path, dst_path)
                        except OSError:
                            # Fallback to copying on Windows
                            shutil.copy2(img_path, dst_path)

                split_stats['split_counts'][split_name] += len(split_images)

            split_stats['valid_monkeys'] += 1

            self.logger.debug(
                f"Monkey {monkey_id}: Train={len(monkey_splits['train'])}, "
                f"Val={len(monkey_splits['val'])}, Test={len(monkey_splits['test'])}"
            )

        # Log statistics
        self.logger.info(f"Dataset split completed:")
        self.logger.info(f"  Valid monkeys: {split_stats['valid_monkeys']}")
        self.logger.info(f"  Skipped monkeys: {len(split_stats['skipped_monkeys'])}")
        self.logger.info(f"  Train images: {split_stats['split_counts']['train']}")
        self.logger.info(f"  Val images: {split_stats['split_counts']['val']}")
        self.logger.info(f"  Test images: {split_stats['split_counts']['test']}")

        return {
            'split_dirs': split_dirs,
            'statistics': split_stats
        }

    def _split_monkey_images(
        self,
        images: List[str],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        min_images_per_split: int
    ) -> Dict[str, List[str]]:
        """Split images for a single monkey.

        Args:
            images: List of image paths for this monkey.
            train_ratio: Training set ratio.
            val_ratio: Validation set ratio.
            test_ratio: Test set ratio.
            min_images_per_split: Minimum images per split.

        Returns:
            Dictionary with split image lists.
        """
        np.random.seed(self.random_state)
        images = np.array(images)
        np.random.shuffle(images)

        n_total = len(images)

        # Calculate split sizes, ensuring minimums
        n_train = max(min_images_per_split, int(n_total * train_ratio))
        n_val = max(min_images_per_split, int(n_total * val_ratio))
        n_test = max(min_images_per_split, n_total - n_train - n_val)

        # Adjust if total exceeds available images
        if n_train + n_val + n_test > n_total:
            # Proportionally reduce while maintaining minimums
            excess = n_train + n_val + n_test - n_total

            # Reduce from largest splits first
            splits = [('train', n_train), ('val', n_val), ('test', n_test)]
            splits.sort(key=lambda x: x[1], reverse=True)

            for split_name, split_size in splits:
                reduction = min(excess, split_size - min_images_per_split)
                if split_name == 'train':
                    n_train -= reduction
                elif split_name == 'val':
                    n_val -= reduction
                else:
                    n_test -= reduction
                excess -= reduction

                if excess <= 0:
                    break

        # Create splits
        train_images = images[:n_train].tolist()
        val_images = images[n_train:n_train + n_val].tolist()
        test_images = images[n_train + n_val:n_train + n_val + n_test].tolist()

        return {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

    def create_stratified_split(
        self,
        source_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        min_samples_per_class: int = 5
    ) -> Dict[str, str]:
        """Create stratified split ensuring balanced representation.

        Args:
            source_dir: Source directory with class subdirectories.
            output_dir: Output directory for split datasets.
            train_ratio: Training set ratio.
            val_ratio: Validation set ratio.
            min_samples_per_class: Minimum samples per class in each split.

        Returns:
            Dictionary with split information.
        """
        if abs(train_ratio + val_ratio - 1.0) > 1e-6:
            raise ValueError("Train and validation ratios must sum to 1.0")

        self.logger.info(f"Creating stratified split from {source_dir}")

        # Collect all samples with their class labels
        all_samples = []
        class_counts = defaultdict(int)

        for class_name in os.listdir(source_dir):
            class_dir = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            images = get_image_files(class_dir)

            if len(images) < min_samples_per_class * 2:
                self.logger.warning(
                    f"Class {class_name} has insufficient samples ({len(images)}), "
                    f"minimum required: {min_samples_per_class * 2}"
                )
                continue

            for img_path in images:
                all_samples.append((img_path, class_name))
                class_counts[class_name] += 1

        if len(all_samples) == 0:
            raise ValueError("No valid samples found for splitting")

        # Extract features and labels
        sample_paths = [s[0] for s in all_samples]
        sample_labels = [s[1] for s in all_samples]

        # Perform stratified split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            sample_paths,
            sample_labels,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=sample_labels
        )

        # Create output directories and copy files
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')

        self._copy_split_files(train_paths, train_labels, train_dir)
        self._copy_split_files(val_paths, val_labels, val_dir)

        # Calculate statistics
        train_class_counts = defaultdict(int)
        val_class_counts = defaultdict(int)

        for label in train_labels:
            train_class_counts[label] += 1
        for label in val_labels:
            val_class_counts[label] += 1

        statistics = {
            'total_classes': len(class_counts),
            'total_samples': len(all_samples),
            'train_samples': len(train_paths),
            'val_samples': len(val_paths),
            'train_class_distribution': dict(train_class_counts),
            'val_class_distribution': dict(val_class_counts)
        }

        self.logger.info(f"Stratified split completed:")
        self.logger.info(f"  Classes: {statistics['total_classes']}")
        self.logger.info(f"  Train samples: {statistics['train_samples']}")
        self.logger.info(f"  Val samples: {statistics['val_samples']}")

        return {
            'train_dir': train_dir,
            'val_dir': val_dir,
            'statistics': statistics
        }

    def _copy_split_files(
        self,
        file_paths: List[str],
        labels: List[str],
        output_dir: str
    ) -> None:
        """Copy files to split directory organized by class.

        Args:
            file_paths: List of file paths to copy.
            labels: Corresponding class labels.
            output_dir: Output directory.
        """
        ensure_dir(output_dir)

        for file_path, label in zip(file_paths, labels):
            # Create class directory
            class_dir = os.path.join(output_dir, label)
            ensure_dir(class_dir)

            # Copy file
            file_name = os.path.basename(file_path)
            dst_path = os.path.join(class_dir, file_name)
            shutil.copy2(file_path, dst_path)

    def analyze_split_balance(self, split_dirs: Dict[str, str]) -> Dict[str, any]:
        """Analyze balance across dataset splits.

        Args:
            split_dirs: Dictionary with split directory paths.

        Returns:
            Analysis results dictionary.
        """
        analysis = {
            'splits': {},
            'class_distribution': defaultdict(dict),
            'balance_metrics': {}
        }

        # Analyze each split
        for split_name, split_dir in split_dirs.items():
            if not os.path.exists(split_dir):
                continue

            split_info = {
                'total_samples': 0,
                'classes': {},
                'class_counts': defaultdict(int)
            }

            # Count samples per class
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue

                images = get_image_files(class_dir)
                count = len(images)

                split_info['classes'][class_name] = count
                split_info['class_counts'][class_name] = count
                split_info['total_samples'] += count

                # Update global class distribution
                analysis['class_distribution'][class_name][split_name] = count

            analysis['splits'][split_name] = split_info

        # Calculate balance metrics
        if len(analysis['splits']) > 1:
            analysis['balance_metrics'] = self._calculate_balance_metrics(
                analysis['class_distribution']
            )

        return analysis

    def _calculate_balance_metrics(
        self,
        class_distribution: Dict[str, Dict[str, int]]
    ) -> Dict[str, float]:
        """Calculate balance metrics across splits.

        Args:
            class_distribution: Class distribution across splits.

        Returns:
            Balance metrics dictionary.
        """
        metrics = {}

        # Calculate coefficient of variation for each class across splits
        class_cvs = []
        for class_name, split_counts in class_distribution.items():
            counts = list(split_counts.values())
            if len(counts) > 1 and np.mean(counts) > 0:
                cv = np.std(counts) / np.mean(counts)
                class_cvs.append(cv)

        if class_cvs:
            metrics['mean_class_cv'] = np.mean(class_cvs)
            metrics['max_class_cv'] = np.max(class_cvs)

        # Calculate overall balance score (lower is better)
        if class_cvs:
            metrics['balance_score'] = np.mean(class_cvs)

        return metrics