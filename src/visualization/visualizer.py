"""Visualization tools for monkey face recognition results."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont

from ..utils.logging import LoggerMixin
from ..utils.data_structures import BoundingBox, MonkeyDetection
from ..utils.error_handler import handle_errors
from ..utils.image_utils import load_image


class ResultVisualizer(LoggerMixin):
    """Visualizer for monkey recognition results."""

    def __init__(self, output_dir: str = "visualizations"):
        """Initialize result visualizer.

        Args:
            output_dir: Directory to save visualization outputs.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color palette for different monkeys
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
        ]

        # Font settings
        self.font_size = 16
        self.font_color = (255, 255, 255)

    @handle_errors(context="drawing bounding boxes", reraise=False)
    def draw_detections_on_image(
        self,
        image: np.ndarray,
        detections: List[MonkeyDetection],
        show_confidence: bool = True,
        show_monkey_id: bool = True
    ) -> np.ndarray:
        """Draw detection results on image.

        Args:
            image: Input image aray.
            detections: List of monkey detections.
            show_confidence: Whether to show confidence scores.
            show_monkey_id: Whether to show monkey IDs.

        Returns:
            Image with drawn detections.
        """
        if len(detections) == 0:
            return image.copy()

        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", self.font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None

        for i, detection in enumerate(detections):
            bbox = detection.bbox

            # Choose color based on monkey ID
            if detection.monkey_id == "unknown":
                color = '#FF0000'  # Red for unknown
            else:
                color_idx = hash(detection.monkey_id) % len(self.colors)
                color = self.colors[color_idx]

            # Convert hex color to RGB
            color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

            # Draw bounding box
            draw.rectangle(
                [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max],
                outline=color_rgb,
                width=3
            )

            # Prepare label text
            label_parts = []
            if show_monkey_id:
                label_parts.append(f"ID: {detection.monkey_id}")
            if show_confidence:
                label_parts.append(f"Conf: {detection.recognition_confidence:.2f}")

            if label_parts:
                label_text = " | ".join(label_parts)

                # Calculate text size and position
                if font:
                    bbox_text = draw.textbbox((0, 0), label_text, font=font)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                else:
                    text_width = len(label_text) * 8
                    text_height = 12

                # Position label above bounding box
                label_x = bbox.x_min
                label_y = max(0, bbox.y_min - text_height - 5)

                # Draw background rectangle for text
                draw.rectangle(
                    [label_x, label_y, label_x + text_width + 10, label_y + text_height + 5],
                    fill=color_rgb,
                    outline=color_rgb
                )

                # Draw text
                draw.text(
                    (label_x + 5, label_y + 2),
                    label_text,
                    fill=self.font_color,
                    font=font
                )

        # Convert back to OpenCV format
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result_image

    @handle_errors(context="saving annotated image", reraise=False)
    def save_annotated_image(
        self,
        image: np.ndarray,
        detections: List[MonkeyDetection],
        output_path: str,
        show_confidence: bool = True,
        show_monkey_id: bool = True
    ) -> bool:
        """Save image with detection annotations.

        Args:
            image: Input image as numpy array.
            detections: List of monkey detections.
            output_path: Path to save annotated image.
            show_confidence: Whether to show confidence scores.
            show_monkey_id: Whether to show monkey IDs.

        Returns:
            True if successful, False otherwise.
        """
        try:
            annotated_image = self.draw_detections_on_image(
                image, detections, show_confidence, show_monkey_id
            )

            success = cv2.imwrite(output_path, annotated_image)
            if success:
                self.logger.info(f"Annotated image saved to {output_path}")
            else:
                self.logger.error(f"Failed to save annotated image to {output_path}")

            return success

        except Exception as e:
            self.logger.error(f"Error saving annotated image: {e}")
            return False

    @handle_errors(context="creating image gallery", reraise=False)
    def create_image_gallery(
        self,
        image_results: List[Dict[str, Any]],
        output_path: str,
        max_images: int = 20,
        grid_size: Tuple[int, int] = None
    ) -> bool:
        """Create an image gallery showing detection results.

        Args:
            image_results: List of dictionaries with 'image_path' and 'detections'.
            output_path: Path to save gallery image.
            max_images: Maximum number of images to include.
            grid_size: Grid size as (rows, cols). Auto-calculated if None.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not image_results:
                self.logger.warning("No image results provided for gallery")
                return False

            # Limit number of images
            image_results = image_results[:max_images]
            num_images = len(image_results)

            # Calculate grid size
            if grid_size is None:
                cols = int(np.ceil(np.sqrt(num_images)))
                rows = int(np.ceil(num_images / cols))
            else:
                rows, cols = grid_size

            # Set up the plot
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for i, result in enumerate(image_results):
                if i >= len(axes):
                    break

                ax = axes[i]

                # Load and process image
                image_path = result['image_path']
                detections = result.get('detections', [])

                image = load_image(image_path)
                if image is None:
                    ax.text(0.5, 0.5, f"Failed to load\\n{Path(image_path).name}",
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # Draw detections
                annotated_image = self.draw_detections_on_image(image, detections)

                # Convert BGR to RGB for matplotlib
                rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                # Display image
                ax.imshow(rgb_image)
                ax.set_title(f"{Path(image_path).name}\\n{len(detections)} faces", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

            # Hide unused subplots
            for i in range(num_images, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Image gallery saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating image gallery: {e}")
            return False

    @handle_errors(context="plotting confusion matrix", reraise=False)
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        output_path: str,
        normalize: bool = False,
        title: str = "Confusion Matrix"
    ) -> bool:
        """Plot confusion matrix.

        Args:
            confusion_matrix: Confusion matrix as numpy array.
            class_names: List of class names.
            output_path: Path to save plot.
            normalize: Whether to normalize the matrix.
            title: Plot title.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if confusion_matrix.size == 0:
                self.logger.warning("Empty confusion matrix provided")
                return False

            # Normalize if requested
            if normalize:
                confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
            else:
                fmt = 'd'

            # Create plot
            plt.figure(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.6)))

            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt=fmt,
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
            )

            plt.title(title, fontsize=16, pad=20)
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Confusion matrix saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {e}")
            return False

    @handle_errors(context="plotting ROC curves", reraise=False)
    def plot_roc_curves(
        self,
        roc_data: Dict[str, Dict[str, np.ndarray]],
        output_path: str,
        title: str = "ROC Curves"
    ) -> bool:
        """Plot ROC curves for multiple classes.

        Args:
            roc_data: Dictionary with class names as keys and 'fpr', 'tpr', 'auc' as values.
            output_path: Path to save plot.
            title: Plot title.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not roc_data:
                self.logger.warning("No ROC data provided")
                return False

            plt.figure(figsize=(10, 8))

            # Plot ROC curve for each class
            for i, (class_name, data) in enumerate(roc_data.items()):
                color = self.colors[i % len(self.colors)]
                plt.plot(
                    data['fpr'],
                    data['tpr'],
                    color=color,
                    lw=2,
                    label=f'{class_name} (AUC = {data["auc"]:.2f})'
                )

            # Plot diagonal line
            plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(title, fontsize=16)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"ROC curves saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error plotting ROC curves: {e}")
            return False

    @handle_errors(context="plotting performance metrics", reraise=False)
    def plot_performance_metrics(
        self,
        metrics_data: Dict[str, Dict[str, float]],
        output_path: str,
        metrics: List[str] = None,
        title: str = "Performance Metrics by Class"
    ) -> bool:
        """Plot performance metrics for different classes.

        Args:
            metrics_data: Dictionary with class names as keys and metrics as values.
            output_path: Path to save plot.
            metrics: List of metrics to plot. Defaults to ['precision', 'recall', 'f1_score'].
            title: Plot title.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not metrics_data:
                self.logger.warning("No metrics data provided")
                return False

            if metrics is None:
                metrics = ['precision', 'recall', 'f1_score']

            classes = list(metrics_data.keys())
            x = np.arange(len(classes))
            width = 0.25

            fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.8), 6))

            # Plot bars for each metric
            for i, metric in enumerate(metrics):
                values = [metrics_data[cls].get(metric, 0) for cls in classes]
                offset = (i - len(metrics) / 2 + 0.5) * width

                bars = ax.bar(
                    x + offset,
                    values,
                    width,
                    label=metric.replace('_', ' ').title(),
                    alpha=0.8
                )

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height + 0.01,
                        f'{value:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )

            ax.set_xlabel('Classes', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(title, fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.1)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Performance metrics plot saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error plotting performance metrics: {e}")
            return False

    @handle_errors(context="plotting confidence analysis", reraise=False)
    def plot_confidence_analysis(
        self,
        confidence_data: Dict[str, Dict[str, float]],
        output_path: str,
        title: str = "Confidence Threshold Analysis"
    ) -> bool:
        """Plot confidence threshold analysis.

        Args:
            confidence_data: Dictionary with threshold keys and metrics values.
            output_path: Path to save plot.
            title: Plot title.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not confidence_data:
                self.logger.warning("No confidence data provided")
                return False

            # Extract data
            thresholds = []
            accuracies = []
            coverages = []

            for key in sorted(confidence_data.keys()):
                threshold = float(key.split('_')[1])
                metrics = confidence_data[key]

                thresholds.append(threshold)
                accuracies.append(metrics.get('accuracy_at_threshold', 0))
                coverages.append(metrics.get('coverage', 0))

            # Create plot with two y-axes
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot accuracy
            color1 = '#1f77b4'
            ax1.set_xlabel('Confidence Threshold', fontsize=12)
            ax1.set_ylabel('Accuracy', color=color1, fontsize=12)
            line1 = ax1.plot(thresholds, accuracies, 'o-', color=color1, linewidth=2, markersize=6, label='Accuracy')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)

            # Create second y-axis for coverage
            ax2 = ax1.twinx()
            color2 = '#ff7f0e'
            ax2.set_ylabel('Coverage', color=color2, fontsize=12)
            line2 = ax2.plot(thresholds, coverages, 's-', color=color2, linewidth=2, markersize=6, label='Coverage')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim(0, 1)

            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='center right')

            plt.title(title, fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Confidence analysis plot saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error plotting confidence analysis: {e}")
            return False

    @handle_errors(context="creating detection summary", reraise=False)
    def create_detection_summary(
        self,
        image_results: List[Dict[str, Any]],
        output_path: str
    ) -> bool:
        """Create a summary visualization of detection results.

        Args:
            image_results: List of dictionaries with detection results.
            output_path: Path to save summary plot.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not image_results:
                self.logger.warning("No image results provided")
                return False

            # Collect statistics
            total_images = len(image_results)
            total_detections = sum(len(result.get('detections', [])) for result in image_results)
            images_with_detections = sum(1 for result in image_results if len(result.get('detections', [])) > 0)

            # Count detections per monkey ID
            monkey_counts = {}
            confidence_scores = []

            for result in image_results:
                for detection in result.get('detections', []):
                    monkey_id = detection.monkey_id
                    monkey_counts[monkey_id] = monkey_counts.get(monkey_id, 0) + 1
                    confidence_scores.append(detection.recognition_confidence)

            # Create summary plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # 1. Overall statistics
            stats_labels = ['Total Images', 'Images with Faces', 'Total Detections']
            stats_values = [total_images, images_with_detections, total_detections]

            bars1 = ax1.bar(stats_labels, stats_values, color=['#3498db', '#2ecc71', '#e74c3c'])
            ax1.set_title('Detection Statistics', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Count')

            # Add value labels on bars
            for bar, value in zip(bars1, stats_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(stats_values) * 0.01,
                        f'{value}', ha='center', va='bottom', fontweight='bold')

            # 2. Detections per monkey
            if monkey_counts:
                sorted_monkeys = sorted(monkey_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                monkey_names = [item[0] for item in sorted_monkeys]
                monkey_values = [item[1] for item in sorted_monkeys]

                bars2 = ax2.bar(range(len(monkey_names)), monkey_values, color='#9b59b6')
                ax2.set_title('Top 10 Detected Monkeys', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Monkey ID')
                ax2.set_ylabel('Detection Count')
                ax2.set_xticks(range(len(monkey_names)))
                ax2.set_xticklabels(monkey_names, rotation=45, ha='right')

                # Add value labels
                for bar, value in zip(bars2, monkey_values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(monkey_values) * 0.01,
                            f'{value}', ha='center', va='bottom')
            else:
                ax2.text(0.5, 0.5, 'No detections found', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Detected Monkeys', fontsize=14, fontweight='bold')

            # 3. Confidence score distribution
            if confidence_scores:
                ax3.hist(confidence_scores, bins=20, color='#f39c12', alpha=0.7, edgecolor='black')
                ax3.set_title('Recognition Confidence Distribution', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Confidence Score')
                ax3.set_ylabel('Frequency')
                ax3.axvline(np.mean(confidence_scores), color='red', linestyle='--',
                           label=f'Mean: {np.mean(confidence_scores):.3f}')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No confidence scores available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Confidence Distribution', fontsize=14, fontweight='bold')

            # 4. Detections per image distribution
            detections_per_image = [len(result.get('detections', [])) for result in image_results]
            unique_counts, count_frequencies = np.unique(detections_per_image, return_counts=True)

            bars4 = ax4.bar(unique_counts, count_frequencies, color='#1abc9c')
            ax4.set_title('Detections per Image Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Number of Detections')
            ax4.set_ylabel('Number of Images')
            ax4.set_xticks(unique_counts)

            # Add value labels
            for bar, value in zip(bars4, count_frequencies):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(count_frequencies) * 0.01,
                        f'{value}', ha='center', va='bottom')

            plt.suptitle('Monkey Face Detection Summary', fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Detection summary saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating detection summary: {e}")
            return False


def create_visualizer(output_dir: str = "visualizations") -> ResultVisualizer:
    """Create result visualizer instance.

    Args:
        output_dir: Directory to save visualization outputs.

    Returns:
        ResultVisualizer instance.
    """
    return ResultVisualizer(output_dir=output_dir)