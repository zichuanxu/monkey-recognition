"""Visualization utilities for the monkey recognition system."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
import os
from pathlib import Path

from .data_structures import BoundingBox, MonkeyDetection
from .image_utils import save_image
from .logging import get_logger

logger = get_logger(__name__)


def draw_bounding_box(
    image: np.ndarray,
    bbox: BoundingBox,
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.7
) -> np.ndarray:
    """Draw bounding box on image.

    Args:
        image: Input image.
        bbox: Bounding box to draw.
        label: Label text to display.
        color: Box color in BGR format.
        thickness: Line thickness.
        font_scale: Font scale for label text.

    Returns:
        Image with bounding box drawn.
    """
    # Create a copy to avoid modifying original
    img_copy = image.copy()

    # Draw rectangle
    cv2.rectangle(
        img_copy,
        (bbox.x_min, bbox.y_min),
        (bbox.x_max, bbox.y_max),
        color,
        thickness
    )

    # Draw label if provided
    if label:
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Draw background rectangle for text
        cv2.rectangle(
            img_copy,
            (bbox.x_min, bbox.y_min - text_height - baseline - 5),
            (bbox.x_min + text_width, bbox.y_min),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            img_copy,
            label,
            (bbox.x_min, bbox.y_min - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )

    return img_copy


def draw_detections(
    image: np.ndarray,
    detections: List[MonkeyDetection],
    show_confidence: bool = True,
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """Draw multiple detections on image.

    Args:
        image: Input image.
        detections: List of monkey detections.
        show_confidence: Whether to show confidence scores in labels.
        color_map: Optional color mapping for different monkey IDs.

    Returns:
        Image with detections drawn.
    """
    img_copy = image.copy()

    # Default colors for different monkeys
    default_colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]

    for i, detection in enumerate(detections):
        # Determine color
        if color_map and detection.monkey_id in color_map:
            color = color_map[detection.monkey_id]
        else:
            color = default_colors[i % len(default_colors)]

        # Create label
        if show_confidence:
            label = f"{detection.monkey_id} ({detection.overall_confidence:.2f})"
        else:
            label = detection.monkey_id

        # Draw bounding box
        img_copy = draw_bounding_box(img_copy, detection.bbox, label, color)

    return img_copy


def create_detection_grid(
    images: List[np.ndarray],
    detections_list: List[List[MonkeyDetection]],
    titles: Optional[List[str]] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """Create a grid of images with detections.

    Args:
        images: List of images.
        detections_list: List of detection lists for each image.
        titles: Optional titles for each image.
        grid_size: Grid size as (rows, cols). If None, auto-calculated.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    n_images = len(images)

    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i in range(n_images):
        # Draw detections on image
        img_with_detections = draw_detections(images[i], detections_list[i])

        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)

        axes[i].imshow(img_rgb)
        axes[i].axis('off')

        if titles and i < len(titles):
            axes[i].set_title(titles[i])

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = False
) -> plt.Figure:
    """Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array.
        class_names: List of class names.
        title: Plot title.
        figsize: Figure size.
        normalize: Whether to normalize the matrix.

    Returns:
        Matplotlib figure.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.tight_layout()
    return fig


def plot_training_metrics(
    metrics_history: List[Dict[str, float]],
    title: str = "Training Metrics",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot training metrics over epochs.

    Args:
        metrics_history: List of metrics dictionaries for each epoch.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    if not metrics_history:
        logger.warning("No metrics history provided")
        return plt.figure()

    epochs = [m.get('epoch', i) for i, m in enumerate(metrics_history)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title)

    # Loss plot
    train_loss = [m.get('train_loss') for m in metrics_history]
    val_loss = [m.get('val_loss') for m in metrics_history]

    axes[0, 0].plot(epochs, train_loss, label='Train Loss', marker='o')
    if any(v is not None for v in val_loss):
        val_loss_clean = [v for v in val_loss if v is not None]
        val_epochs = [e for e, v in zip(epochs, val_loss) if v is not None]
        axes[0, 0].plot(val_epochs, val_loss_clean, label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy plot
    train_acc = [m.get('train_accuracy') for m in metrics_history]
    val_acc = [m.get('val_accuracy') for m in metrics_history]

    if any(v is not None for v in train_acc):
        train_acc_clean = [v for v in train_acc if v is not None]
        train_epochs = [e for e, v in zip(epochs, train_acc) if v is not None]
        axes[0, 1].plot(train_epochs, train_acc_clean, label='Train Accuracy', marker='o')

    if any(v is not None for v in val_acc):
        val_acc_clean = [v for v in val_acc if v is not None]
        val_epochs = [e for e, v in zip(epochs, val_acc) if v is not None]
        axes[0, 1].plot(val_epochs, val_acc_clean, label='Val Accuracy', marker='s')

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning rate plot
    learning_rates = [m.get('learning_rate') for m in metrics_history]
    if any(v is not None for v in learning_rates):
        lr_clean = [v for v in learning_rates if v is not None]
        lr_epochs = [e for e, v in zip(epochs, learning_rates) if v is not None]
        axes[1, 0].plot(lr_epochs, lr_clean, marker='o')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Learning Rate Data',
                       ha='center', va='center', transform=axes[1, 0].transAxes)

    # Combined loss and accuracy
    axes[1, 1].plot(epochs, train_loss, label='Train Loss', marker='o', alpha=0.7)
    if any(v is not None for v in val_loss):
        axes[1, 1].plot(val_epochs, val_loss_clean, label='Val Loss', marker='s', alpha=0.7)

    # Create second y-axis for accuracy
    ax2 = axes[1, 1].twinx()
    if any(v is not None for v in train_acc):
        ax2.plot(train_epochs, train_acc_clean, label='Train Acc', marker='^', color='green', alpha=0.7)
    if any(v is not None for v in val_acc):
        ax2.plot(val_epochs, val_acc_clean, label='Val Acc', marker='v', color='red', alpha=0.7)

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    axes[1, 1].set_title('Loss and Accuracy')

    # Combine legends
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='center right')

    axes[1, 1].grid(True)

    plt.tight_layout()
    return fig


def save_visualization(
    fig: plt.Figure,
    save_path: str,
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> bool:
    """Save matplotlib figure to file.

    Args:
        fig: Matplotlib figure.
        save_path: Path to save the figure.
        dpi: Resolution in dots per inch.
        bbox_inches: Bounding box setting.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)  # Close figure to free memory
        return True
    except Exception as e:
        logger.error(f"Error saving visualization to {save_path}: {e}")
        return False


def create_detection_summary_image(
    image: np.ndarray,
    detections: List[MonkeyDetection],
    save_path: Optional[str] = None
) -> np.ndarray:
    """Create a summary image with detection statistics.

    Args:
        image: Input image.
        detections: List of detections.
        save_path: Optional path to save the image.

    Returns:
        Summary image with statistics.
    """
    # Draw detections on image
    img_with_detections = draw_detections(image, detections)

    # Add statistics text
    h, w = img_with_detections.shape[:2]

    # Create text overlay
    stats_text = [
        f"Total Detections: {len(detections)}",
        f"Unique Monkeys: {len(set(d.monkey_id for d in detections))}",
    ]

    # Add individual detection info
    for i, detection in enumerate(detections):
        stats_text.append(
            f"{i+1}. {detection.monkey_id} "
            f"(conf: {detection.overall_confidence:.2f})"
        )

    # Draw text background
    text_height = 25
    text_bg_height = len(stats_text) * text_height + 20

    # Create text background
    cv2.rectangle(
        img_with_detections,
        (10, 10),
        (400, text_bg_height),
        (0, 0, 0),
        -1
    )

    # Draw text
    for i, text in enumerate(stats_text):
        cv2.putText(
            img_with_detections,
            text,
            (20, 35 + i * text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

    # Save if path provided
    if save_path:
        save_image(img_with_detections, save_path)

    return img_with_detections