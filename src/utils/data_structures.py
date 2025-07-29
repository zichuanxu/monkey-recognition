"""Core data structures for the monkey recognition system."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class BoundingBox:
    """Represents a bounding box for object detection."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float = 0.0

    @property
    def width(self) -> int:
        """Get bounding box width."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Get bounding box height."""
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        """Get bounding box area."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Get bounding box center coordinates."""
        return (
            (self.x_min + self.x_max) // 2,
            (self.y_min + self.y_max) // 2
        )

    def to_yolo_format(self, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
        """Convert to YOLO format (normalized center coordinates and dimensions).

        Args:
            image_width: Image width in pixels.
            image_height: Image height in pixels.

        Returns:
            Tuple of (x_center_norm, y_center_norm, width_norm, height_norm).
        """
        x_center = (self.x_min + self.x_max) / 2.0
        y_center = (self.y_min + self.y_max) / 2.0

        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        width_norm = self.width / image_width
        height_norm = self.height / image_height

        return x_center_norm, y_center_norm, width_norm, height_norm

    @classmethod
    def from_yolo_format(
        cls,
        x_center_norm: float,
        y_center_norm: float,
        width_norm: float,
        height_norm: float,
        image_width: int,
        image_height: int,
        confidence: float = 0.0
    ) -> 'BoundingBox':
        """Create BoundingBox from YOLO format.

        Args:
            x_center_norm: Normalized x center coordinate.
            y_center_norm: Normalized y center coordinate.
            width_norm: Normalized width.
            height_norm: Normalized height.
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            confidence: Detection confidence.

        Returns:
            BoundingBox instance.
        """
        x_center = x_center_norm * image_width
        y_center = y_center_norm * image_height
        width = width_norm * image_width
        height = height_norm * image_height

        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        return cls(x_min, y_min, x_max, y_max, confidence)

    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box.

        Args:
            other: Another BoundingBox instance.

        Returns:
            IoU value between 0 and 1.
        """
        # Calculate intersection coordinates
        x_min_inter = max(self.x_min, other.x_min)
        y_min_inter = max(self.y_min, other.y_min)
        x_max_inter = min(self.x_max, other.x_max)
        y_max_inter = min(self.y_max, other.y_max)

        # Check if there's no intersection
        if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
            return 0.0

        # Calculate intersection area
        intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)

        # Calculate union area
        union_area = self.area + other.area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0


@dataclass
class MonkeyDetection:
    """Represents a detected monkey with identification information."""
    bbox: BoundingBox
    monkey_id: str
    recognition_confidence: float
    detection_confidence: float
    features: Optional[np.ndarray] = None

    @property
    def overall_confidence(self) -> float:
        """Get overall confidence as product of detection and recognition confidence."""
        return self.detection_confidence * self.recognition_confidence


@dataclass
class TrainingMetrics:
    """Training metrics for model evaluation."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy,
            'learning_rate': self.learning_rate
        }


@dataclass
class EvaluationResults:
    """Results from model evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Optional[np.ndarray] = None
    class_names: Optional[List[str]] = None
    predictions: Optional[List[str]] = None
    ground_truth: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving/logging."""
        result = {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score
        }

        if self.class_names is not None:
            result['class_names'] = self.class_names
        if self.predictions is not None:
            result['predictions'] = self.predictions
        if self.ground_truth is not None:
            result['ground_truth'] = self.ground_truth

        return result


@dataclass
class ModelInfo:
    """Information about a trained model."""
    model_path: str
    model_type: str  # 'detection' or 'recognition'
    architecture: str
    training_config: Dict[str, Any]
    performance_metrics: Optional[EvaluationResults] = None
    creation_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving."""
        result = {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'architecture': self.architecture,
            'training_config': self.training_config,
            'creation_date': self.creation_date
        }

        if self.performance_metrics is not None:
            result['performance_metrics'] = self.performance_metrics.to_dict()

        return result