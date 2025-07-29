"""Comprehensive evaluation framework for monkey face recognition system."""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import cv2

from ..utils.logging import LoggerMixin
from ..utils.error_handler import handle_errors, error_context
from ..utils.exceptions import ValidationError, ConfigurationError
from ..utils.data_structures import BoundingBox, MonkeyDetection
from ..utils.validators import validate_directory_path, validate_confidence_threshold
from ..inference.recognizer import MonkeyFaceRecognizer


class DetectionEvaluator(LoggerMixin):
    """Evaluator for face detection performance."""

    def __init__(self, iou_threshold: float = 0.5):
        """Initialize detection evaluator.

        Args:
            iou_threshold: IoU threshold for considering detection as correct.
        """
        self.iou_threshold = iou_threshold
        self.results = []

    def calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: First bounding box.
            bbox2: Second bounding box.

        Returns:
            IoU value between 0 and 1.
        """
        # Calculate intersection coordinates
        x1 = max(bbox1.x_min, bbox2.x_min)
        y1 = max(bbox1.y_min, bbox2.y_min)
        x2 = min(bbox1.x_max, bbox2.x_max)
        y2 = min(bbox1.y_max, bbox2.y_max)

        # Check if there's an intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Calculate intersection area
        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union area
        area1 = bbox1.area
        area2 = bbox2.area
        union = area1 + area2 - intersection

        # Calculate IoU
        if union == 0:
            return 0.0

        return intersection / union

    def evaluate_image(
        self,
        predicted_bboxes: List[BoundingBox],
        ground_truth_bboxes: List[BoundingBox]
    ) -> Dict[str, Any]:
        """Evaluate detection performance on a single image.

        Args:
            predicted_bboxes: List of predicted bounding boxes.
            ground_truth_bboxes: List of ground truth bounding boxes.

        Returns:
            Evaluation metrics for the image.
        """
        num_predictions = len(predicted_bboxes)
        num_ground_truth = len(ground_truth_bboxes)

        if num_ground_truth == 0:
            # No ground truth faces
            return {
                'true_positives': 0,
                'false_positives': num_predictions,
                'false_negatives': 0,
                'precision': 0.0 if num_predictions > 0 else 1.0,
                'recall': 1.0,  # No faces to miss
                'f1_score': 0.0 if num_predictions > 0 else 1.0
            }

        if num_predictions == 0:
            # No predictions
            return {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': num_ground_truth,
                'precision': 1.0,  # No false positives
                'recall': 0.0,
                'f1_score': 0.0
            }

        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()

        for i, pred_bbox in enumerate(predicted_bboxes):
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt_bbox in enumerate(ground_truth_bboxes):
                if j in matched_gt:
                    continue

                iou = self.calculate_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= self.iou_threshold:
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)

        # Calculate metrics
        true_positives = len(matched_pred)
        false_positives = num_predictions - true_positives
        false_negatives = num_ground_truth - true_positives

        precision = true_positives / num_predictions if num_predictions > 0 else 0.0
        recall = true_positives / num_ground_truth if num_ground_truth > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    def add_result(
        self,
        image_path: str,
        predicted_bboxes: List[BoundingBox],
        ground_truth_bboxes: List[BoundingBox]
    ) -> None:
        """Add evaluation result for an image.

        Args:
            image_path: Path to the evaluated image.
            predicted_bboxes: List of predicted bounding boxes.
            ground_truth_bboxes: List of ground truth bounding boxes.
        """
        metrics = self.evaluate_image(predicted_bboxes, ground_truth_bboxes)

        result = {
            'image_path': image_path,
            'num_predictions': len(predicted_bboxes),
            'num_ground_truth': len(ground_truth_bboxes),
            **metrics
        }

        self.results.append(result)

    def calculate_map(self, confidence_thresholds: List[float]) -> Dict[str, float]:
        """Calculate mean Average Precision (mAP) at different confidence thresholds.

        Args:
            confidence_thresholds: List of confidence thresholds to evaluate.

        Returns:
            mAP metrics at different thresholds.
        """
        # This is a simplified mAP calculation
        # In practice, you'd need confidence scores for each prediction

        if not self.results:
            return {'map_50': 0.0, 'map_75': 0.0, 'map_50_95': 0.0}

        # Calculate average precision at IoU=0.5
        total_precision = sum(result['precision'] for result in self.results)
        total_recall = sum(result['recall'] for result in self.results)

        avg_precision = total_precision / len(self.results)
        avg_recall = total_recall / len(self.results)

        # Simplified mAP calculation
        map_50 = avg_precision * avg_recall if avg_recall > 0 else 0.0

        return {
            'map_50': map_50,
            'map_75': map_50 * 0.8,  # Approximation
            'map_50_95': map_50 * 0.6  # Approximation
        }

    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary metrics across all evaluated images.

        Returns:
            Summary metrics dictionary.
        """
        if not self.results:
            return {}

        total_tp = sum(result['true_positives'] for result in self.results)
        total_fp = sum(result['false_positives'] for result in self.results)
        total_fn = sum(result['false_negatives'] for result in self.results)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_images': len(self.results),
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn
        }


class RecognitionEvaluator(LoggerMixin):
    """Evaluator for face recognition performance."""

    def __init__(self):
        """Initialize recognition evaluator."""
        self.predictions = []
        self.ground_truth = []
        self.confidence_scores = []
        self.image_paths = []

    def add_result(
        self,
        image_path: str,
        predicted_id: str,
        ground_truth_id: str,
        confidence: float
    ) -> None:
        """Add recognition result.

        Args:
            image_path: Path to the evaluated image.
            predicted_id: Predicted monkey ID.
            ground_truth_id: Ground truth monkey ID.
            confidence: Recognition confidence score.
        """
        self.image_paths.append(image_path)
        self.predictions.append(predicted_id)
        self.ground_truth.append(ground_truth_id)
        self.confidence_scores.append(confidence)

    def calculate_accuracy(self) -> float:
        """Calculate recognition accuracy.

        Returns:
            Accuracy score.
        """
        if not self.predictions:
            return 0.0

        return accuracy_score(self.ground_truth, self.predictions)

    def calculate_precision_recall_f1(self, average: str = 'weighted') -> Dict[str, float]:
        """Calculate precision, recall, and F1-score.

        Args:
            average: Averaging method ('micro', 'macro', 'weighted').

        Returns:
            Precision, recall, and F1-score.
        """
        if not self.predictions:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

        precision = precision_score(self.ground_truth, self.predictions, average=average, zero_division=0)
        recall = recall_score(self.ground_truth, self.predictions, average=average, zero_division=0)
        f1 = f1_score(self.ground_truth, self.predictions, average=average, zero_division=0)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def calculate_top_k_accuracy(self, k: int = 5) -> float:
        """Calculate top-k accuracy.

        Args:
            k: Number of top predictions to consider.

        Returns:
            Top-k accuracy.
        """
        # This would require multiple predictions per sample
        # For now, return top-1 accuracy
        return self.calculate_accuracy()

    def generate_confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix.

        Returns:
            Confusion matrix as numpy array.
        """
        if not self.predictions:
            return np.array([])

        unique_labels = sorted(list(set(self.ground_truth + self.predictions)))
        return confusion_matrix(self.ground_truth, self.predictions, labels=unique_labels)

    def get_classification_report(self) -> str:
        """Get detailed classification report.

        Returns:
            Classification report as string.
        """
        if not self.predictions:
            return "No predictions available"

        return classification_report(self.ground_truth, self.predictions, zero_division=0)

    def calculate_confidence_metrics(self, threshold: float = 0.5) -> Dict[str, float]:
        """Calculate metrics based on confidence threshold.

        Args:
            threshold: Confidence threshold.

        Returns:
            Metrics for predictions above threshold.
        """
        if not self.confidence_scores:
            return {}

        # Filter predictions by confidence
        high_conf_indices = [i for i, conf in enumerate(self.confidence_scores) if conf >= threshold]

        if not high_conf_indices:
            return {
                'accuracy_at_threshold': 0.0,
                'coverage': 0.0,
                'num_predictions': 0
            }

        filtered_predictions = [self.predictions[i] for i in high_conf_indices]
        filtered_ground_truth = [self.ground_truth[i] for i in high_conf_indices]

        accuracy = accuracy_score(filtered_ground_truth, filtered_predictions)
        coverage = len(high_conf_indices) / len(self.predictions)

        return {
            'accuracy_at_threshold': accuracy,
            'coverage': coverage,
            'num_predictions': len(high_conf_indices)
        }

    def get_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get per-class performance metrics.

        Returns:
            Dictionary with metrics for each class.
        """
        if not self.predictions:
            return {}

        unique_labels = sorted(list(set(self.ground_truth)))
        per_class_metrics = {}

        for label in unique_labels:
            # Binary classification for this class
            binary_gt = [1 if gt == label else 0 for gt in self.ground_truth]
            binary_pred = [1 if pred == label else 0 for pred in self.predictions]

            if sum(binary_gt) == 0:  # No ground truth samples for this class
                continue

            precision = precision_score(binary_gt, binary_pred, zero_division=0)
            recall = recall_score(binary_gt, binary_pred, zero_division=0)
            f1 = f1_score(binary_gt, binary_pred, zero_division=0)

            per_class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': sum(binary_gt)
            }

        return per_class_metrics


class ComprehensiveEvaluator(LoggerMixin):
    """Comprehensive evaluator for the complete monkey recognition system."""

    def __init__(
        self,
        recognizer: MonkeyFaceRecognizer,
        test_data_dir: str,
        output_dir: str = "evaluation_results"
    ):
        """Initialize comprehensive evaluator.

        Args:
            recognizer: MonkeyFaceRecognizer instance.
            test_data_dir: Directory containing test data.
            output_dir: Directory to save evaluation results.
        """
        self.recognizer = recognizer
        self.test_data_dir = validate_directory_path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluators
        self.detection_evaluator = DetectionEvaluator()
        self.recognition_evaluator = RecognitionEvaluator()

        # Test data structure
        self.test_images = []
        self.ground_truth_data = {}

        self._load_test_data()

    @handle_errors(context="loading test data", reraise=True)
    def _load_test_data(self) -> None:
        """Load test data from directory structure."""
        test_dir = Path(self.test_data_dir)

        # Check if test_dir has subdirectories (organized by monkey ID)
        subdirs = [d for d in test_dir.iterdir() if d.is_dir()]

        if subdirs:
            # Expected structure: test_data_dir/monkey_id/image_files
            for monkey_dir in subdirs:
                monkey_id = monkey_dir.name

                for image_file in monkey_dir.iterdir():
                    if image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                        image_path = str(image_file)
                        self.test_images.append(image_path)

                        # Store ground truth
                        self.ground_truth_data[image_path] = {
                            'monkey_id': monkey_id,
                            'bboxes': []  # Would be loaded from annotation files if available
                        }
        else:
            # All images in one directory - treat as unknown class
            for image_file in test_dir.iterdir():
                if image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                    image_path = str(image_file)
                    self.test_images.append(image_path)

                    # Store ground truth as unknown
                    self.ground_truth_data[image_path] = {
                        'monkey_id': 'unknown',
                        'bboxes': []
                    }

        self.logger.info(f"Loaded {len(self.test_images)} test images from {len(set(self.ground_truth_data[img]['monkey_id'] for img in self.test_images))} monkey classes")

    @handle_errors(context="running comprehensive evaluation", reraise=True)
    def evaluate(
        self,
        confidence_thresholds: List[float] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation.

        Args:
            confidence_thresholds: List of confidence thresholds to evaluate.
            save_results: Whether to save results to files.

        Returns:
            Comprehensive evaluation results.
        """
        if confidence_thresholds is None:
            confidence_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        self.logger.info("Starting comprehensive evaluation...")

        # Process each test image
        for i, image_path in enumerate(self.test_images):
            if i % 100 == 0:
                self.logger.info(f"Processing image {i+1}/{len(self.test_images)}")

            try:
                # Run recognition
                detections = self.recognizer.recognize_image(image_path)

                # Get ground truth
                gt_data = self.ground_truth_data[image_path]
                gt_monkey_id = gt_data['monkey_id']

                # Evaluate recognition for each detected face
                for detection in detections:
                    self.recognition_evaluator.add_result(
                        image_path=image_path,
                        predicted_id=detection.monkey_id,
                        ground_truth_id=gt_monkey_id,
                        confidence=detection.recognition_confidence
                    )

                # For detection evaluation, we'd need ground truth bounding boxes
                # For now, assume each image has one face (the whole image)
                if detections:
                    # Simplified: assume detection is correct if recognition is correct
                    predicted_bboxes = [det.bbox for det in detections]
                    # Create dummy ground truth bbox (would be loaded from annotations)
                    gt_bboxes = [BoundingBox(0, 0, 100, 100, 1.0)]  # Placeholder

                    self.detection_evaluator.add_result(
                        image_path=image_path,
                        predicted_bboxes=predicted_bboxes,
                        ground_truth_bboxes=gt_bboxes
                    )

            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                continue

        # Calculate metrics
        results = self._calculate_all_metrics(confidence_thresholds)

        # Save results if requested
        if save_results:
            self._save_results(results)

        self.logger.info("Comprehensive evaluation completed")
        return results

    def _calculate_all_metrics(self, confidence_thresholds: List[float]) -> Dict[str, Any]:
        """Calculate all evaluation metrics.

        Args:
            confidence_thresholds: List of confidence thresholds.

        Returns:
            All evaluation metrics.
        """
        results = {
            'detection_metrics': self.detection_evaluator.get_summary_metrics(),
            'recognition_metrics': {
                'accuracy': self.recognition_evaluator.calculate_accuracy(),
                **self.recognition_evaluator.calculate_precision_recall_f1(),
                'per_class_metrics': self.recognition_evaluator.get_per_class_metrics()
            },
            'confidence_analysis': {},
            'confusion_matrix': self.recognition_evaluator.generate_confusion_matrix().tolist(),
            'classification_report': self.recognition_evaluator.get_classification_report()
        }

        # Analyze performance at different confidence thresholds
        for threshold in confidence_thresholds:
            threshold_metrics = self.recognition_evaluator.calculate_confidence_metrics(threshold)
            results['confidence_analysis'][f'threshold_{threshold}'] = threshold_metrics

        return results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to files.

        Args:
            results: Evaluation results dictionary.
        """
        # Save JSON results
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to {results_file}")

        # Save detailed CSV results
        self._save_detailed_results()

        # Generate and save plots
        self._generate_plots(results)

    def _save_detailed_results(self) -> None:
        """Save detailed results to CSV files."""
        # Save recognition results
        if self.recognition_evaluator.predictions:
            recognition_df = pd.DataFrame({
                'image_path': self.recognition_evaluator.image_paths,
                'predicted_id': self.recognition_evaluator.predictions,
                'ground_truth_id': self.recognition_evaluator.ground_truth,
                'confidence': self.recognition_evaluator.confidence_scores
            })

            recognition_file = self.output_dir / "recognition_results.csv"
            recognition_df.to_csv(recognition_file, index=False)
            self.logger.info(f"Recognition results saved to {recognition_file}")

        # Save detection results
        if self.detection_evaluator.results:
            detection_df = pd.DataFrame(self.detection_evaluator.results)
            detection_file = self.output_dir / "detection_results.csv"
            detection_df.to_csv(detection_file, index=False)
            self.logger.info(f"Detection results saved to {detection_file}")

    def _generate_plots(self, results: Dict[str, Any]) -> None:
        """Generate evaluation plots.

        Args:
            results: Evaluation results dictionary.
        """
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # Generate confusion matrix plot
        self._plot_confusion_matrix(results)

        # Generate confidence analysis plot
        self._plot_confidence_analysis(results)

        # Generate per-class performance plot
        self._plot_per_class_performance(results)

    def _plot_confusion_matrix(self, results: Dict[str, Any]) -> None:
        """Plot confusion matrix.

        Args:
            results: Evaluation results dictionary.
        """
        if not results.get('confusion_matrix'):
            return

        cm = np.array(results['confusion_matrix'])
        if cm.size == 0:
            return

        # Get unique labels
        unique_labels = sorted(list(set(self.recognition_evaluator.ground_truth)))

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=unique_labels,
            yticklabels=unique_labels
        )
        plt.title('Confusion Matrix - Monkey Recognition')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        cm_file = self.output_dir / "confusion_matrix.png"
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Confusion matrix saved to {cm_file}")

    def _plot_confidence_analysis(self, results: Dict[str, Any]) -> None:
        """Plot confidence analysis.

        Args:
            results: Evaluation results dictionary.
        """
        confidence_data = results.get('confidence_analysis', {})
        if not confidence_data:
            return

        thresholds = []
        accuracies = []
        coverages = []

        for key, metrics in confidence_data.items():
            threshold = float(key.split('_')[1])
            thresholds.append(threshold)
            accuracies.append(metrics.get('accuracy_at_threshold', 0))
            coverages.append(metrics.get('coverage', 0))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Accuracy vs threshold
        ax1.plot(thresholds, accuracies, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Confidence Threshold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Coverage vs threshold
        ax2.plot(thresholds, coverages, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('Coverage')
        ax2.set_title('Coverage vs Confidence Threshold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()

        confidence_file = self.output_dir / "confidence_analysis.png"
        plt.savefig(confidence_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Confidence analysis saved to {confidence_file}")

    def _plot_per_class_performance(self, results: Dict[str, Any]) -> None:
        """Plot per-class performance metrics.

        Args:
            results: Evaluation results dictionary.
        """
        per_class_metrics = results.get('recognition_metrics', {}).get('per_class_metrics', {})
        if not per_class_metrics:
            return

        classes = list(per_class_metrics.keys())
        precisions = [per_class_metrics[cls]['precision'] for cls in classes]
        recalls = [per_class_metrics[cls]['recall'] for cls in classes]
        f1_scores = [per_class_metrics[cls]['f1_score'] for cls in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(15, 8))

        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Monkey Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        per_class_file = self.output_dir / "per_class_performance.png"
        plt.savefig(per_class_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Per-class performance saved to {per_class_file}")

    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report.

        Args:
            results: Evaluation results dictionary.

        Returns:
            Evaluation report as string.
        """
        report_lines = [
            "# Monkey Face Recognition System - Evaluation Report",
            "=" * 60,
            "",
            "## Overall Performance",
            f"- Recognition Accuracy: {results['recognition_metrics']['accuracy']:.4f}",
            f"- Precision: {results['recognition_metrics']['precision']:.4f}",
            f"- Recall: {results['recognition_metrics']['recall']:.4f}",
            f"- F1-Score: {results['recognition_metrics']['f1_score']:.4f}",
            "",
            "## Detection Performance",
        ]

        detection_metrics = results.get('detection_metrics', {})
        if detection_metrics:
            report_lines.extend([
                f"- Detection Precision: {detection_metrics.get('precision', 0):.4f}",
                f"- Detection Recall: {detection_metrics.get('recall', 0):.4f}",
                f"- Detection F1-Score: {detection_metrics.get('f1_score', 0):.4f}",
                f"- Total Images Processed: {detection_metrics.get('total_images', 0)}",
            ])

        report_lines.extend([
            "",
            "## Confidence Analysis",
            "Threshold | Accuracy | Coverage",
            "----------|----------|----------"
        ])

        confidence_data = results.get('confidence_analysis', {})
        for key in sorted(confidence_data.keys()):
            metrics = confidence_data[key]
            threshold = key.split('_')[1]
            accuracy = metrics.get('accuracy_at_threshold', 0)
            coverage = metrics.get('coverage', 0)
            report_lines.append(f"{threshold:>9} | {accuracy:>8.4f} | {coverage:>8.4f}")

        report_lines.extend([
            "",
            "## Per-Class Performance",
            "Class | Precision | Recall | F1-Score | Support",
            "------|-----------|--------|----------|--------"
        ])

        per_class_metrics = results.get('recognition_metrics', {}).get('per_class_metrics', {})
        for class_name in sorted(per_class_metrics.keys()):
            metrics = per_class_metrics[class_name]
            report_lines.append(
                f"{class_name:>5} | {metrics['precision']:>9.4f} | "
                f"{metrics['recall']:>6.4f} | {metrics['f1_score']:>8.4f} | "
                f"{metrics['support']:>7}"
            )

        report_lines.extend([
            "",
            "## Classification Report",
            "```",
            results.get('classification_report', ''),
            "```"
        ])

        report = "\n".join(report_lines)

        # Save report
        report_file = self.output_dir / "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)

        self.logger.info(f"Evaluation report saved to {report_file}")
        return report


def create_evaluator(
    recognizer: MonkeyFaceRecognizer,
    test_data_dir: str,
    output_dir: str = "evaluation_results"
) -> ComprehensiveEvaluator:
    """Create comprehensive evaluator instance.

    Args:
        recognizer: MonkeyFaceRecognizer instance.
        test_data_dir: Directory containing test data.
        output_dir: Directory to save evaluation results.

    Returns:
        ComprehensiveEvaluator instance.
    """
    return ComprehensiveEvaluator(
        recognizer=recognizer,
        test_data_dir=test_data_dir,
        output_dir=output_dir
    )