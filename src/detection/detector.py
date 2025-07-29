"""YOLOv8-based monkey face detector."""

import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple, Union, Dict, Any
from ultralytics import YOLO
from pathlib import Path
import os

from ..utils.data_structures import BoundingBox, MonkeyDetection
from ..utils.image_utils import load_image, resize_image
from ..utils.logging import LoggerMixin


class MonkeyFaceDetector(LoggerMixin):
    """YOLOv8-based monkey face detector."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        device: str = 'auto'
    ):
        """Initialize monkey face detector.

        Args:
            model_path: Path to trained YOLO model. If None, uses pre-trained model.
            confidence_threshold: Confidence threshold for detections.
            iou_threshold: IoU threshold for NMS.
            max_detections: Maximum number of detections per image.
            device: Device to run inference on ('auto', 'cpu', 'cuda').
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = self._setup_device(device)

        # Load model
        self.model = None
        self.model_path = model_path
        self.load_model(model_path)

        self.logger.info(f"MonkeyFaceDetector initialized with device: {self.device}")

    def _setup_device(self, device: str) -> str:
        """Setup computation device.

        Args:
            device: Device specification.

        Returns:
            Actual device to use.
        """
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load YOLO model.

        Args:
            model_path: Path to model file. If None, loads default pre-trained model.
        """
        try:
            if model_path is None or not os.path.exists(model_path):
                # Load pre-trained YOLOv8 model
                self.logger.info("Loading pre-trained YOLOv8 model")
                self.model = YOLO('yolov8m.pt')
                self.model_path = 'yolov8m.pt'
            else:
                # Load custom trained model
                self.logger.info(f"Loading custom model from {model_path}")
                self.model = YOLO(model_path)
                self.model_path = model_path

            # Move model to device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')

            self.logger.info(f"Model loaded successfully: {self.model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def detect_faces(
        self,
        image: Union[str, np.ndarray],
        return_crops: bool = False,
        visualize: bool = False
    ) -> List[BoundingBox]:
        """Detect monkey faces in image.

        Args:
            image: Input image (path or numpy array).
            return_crops: Whether to return cropped face images.
            visualize: Whether to return visualization.

        Returns:
            List of detected face bounding boxes.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load image if path provided
        if isinstance(image, str):
            img_array = load_image(image)
            if img_array is None:
                self.logger.error(f"Failed to load image: {image}")
                return []
        else:
            img_array = image.copy()

        try:
            # Run inference
            results = self.model(
                img_array,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )

            # Parse results
            detections = self._parse_yolo_results(results[0], img_array.shape[:2])

            self.logger.debug(f"Detected {len(detections)} faces")
            return detections

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []

    def detect_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 8
    ) -> List[List[BoundingBox]]:
        """Detect faces in batch of images.

        Args:
            images: List of images (paths or numpy arrays).
            batch_size: Batch size for processing.

        Returns:
            List of detection lists for each image.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        all_detections = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_arrays = []

            # Load images in batch
            for img in batch:
                if isinstance(img, str):
                    img_array = load_image(img)
                    if img_array is None:
                        self.logger.warning(f"Failed to load image: {img}")
                        batch_arrays.append(None)
                    else:
                        batch_arrays.append(img_array)
                else:
                    batch_arrays.append(img)

            # Run batch inference
            try:
                valid_images = [img for img in batch_arrays if img is not None]
                if len(valid_images) == 0:
                    all_detections.extend([[] for _ in batch])
                    continue

                results = self.model(
                    valid_images,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_detections,
                    verbose=False
                )

                # Parse results for each image
                valid_idx = 0
                for img_array in batch_arrays:
                    if img_array is None:
                        all_detections.append([])
                    else:
                        detections = self._parse_yolo_results(
                            results[valid_idx], img_array.shape[:2]
                        )
                        all_detections.append(detections)
                        valid_idx += 1

            except Exception as e:
                self.logger.error(f"Batch detection failed: {e}")
                all_detections.extend([[] for _ in batch])

        return all_detections

    def _parse_yolo_results(
        self,
        result,
        image_shape: Tuple[int, int]
    ) -> List[BoundingBox]:
        """Parse YOLO detection results.

        Args:
            result: YOLO result object.
            image_shape: Image shape (height, width).

        Returns:
            List of BoundingBox objects.
        """
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # Extract bounding boxes and confidences
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()

        # Convert to BoundingBox objects
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box

            # Ensure coordinates are within image bounds
            h, w = image_shape
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(x1 + 1, min(int(x2), w))
            y2 = max(y1 + 1, min(int(y2), h))

            bbox = BoundingBox(x1, y1, x2, y2, float(conf))
            detections.append(bbox)

        return detections

    def crop_faces(
        self,
        image: np.ndarray,
        detections: List[BoundingBox],
        padding: int = 10,
        target_size: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """Crop detected faces from image.

        Args:
            image: Input image.
            detections: List of face detections.
            padding: Padding around face crop.
            target_size: Optional target size for crops.

        Returns:
            List of cropped face images.
        """
        crops = []
        h, w = image.shape[:2]

        for detection in detections:
            # Add padding
            x1 = max(0, detection.x_min - padding)
            y1 = max(0, detection.y_min - padding)
            x2 = min(w, detection.x_max + padding)
            y2 = min(h, detection.y_max + padding)

            # Crop face
            face_crop = image[y1:y2, x1:x2]

            # Resize if target size specified
            if target_size is not None:
                face_crop = resize_image(face_crop, target_size, maintain_aspect_ratio=False)

            crops.append(face_crop)

        return crops

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[BoundingBox],
        show_confidence: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Visualize detections on image.

        Args:
            image: Input image.
            detections: List of detections.
            show_confidence: Whether to show confidence scores.
            color: Bounding box color.
            thickness: Line thickness.

        Returns:
            Image with visualized detections.
        """
        vis_image = image.copy()

        for detection in detections:
            # Draw bounding box
            cv2.rectangle(
                vis_image,
                (detection.x_min, detection.y_min),
                (detection.x_max, detection.y_max),
                color,
                thickness
            )

            # Draw confidence score
            if show_confidence:
                label = f"Face: {detection.confidence:.2f}"

                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness
                )

                # Draw background rectangle
                cv2.rectangle(
                    vis_image,
                    (detection.x_min, detection.y_min - text_height - baseline - 5),
                    (detection.x_min + text_width, detection.y_min),
                    color,
                    -1
                )

                # Draw text
                cv2.putText(
                    vis_image,
                    label,
                    (detection.x_min, detection.y_min - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    thickness
                )

        return vis_image

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information.
        """
        if self.model is None:
            return {'loaded': False}

        info = {
            'loaded': True,
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'max_detections': self.max_detections
        }

        # Add model-specific info if available
        try:
            if hasattr(self.model, 'model'):
                info['model_type'] = str(type(self.model.model))
                if hasattr(self.model.model, 'names'):
                    info['class_names'] = self.model.model.names
        except:
            pass

        return info

    def update_thresholds(
        self,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None
    ) -> None:
        """Update detection thresholds.

        Args:
            confidence_threshold: New confidence threshold.
            iou_threshold: New IoU threshold.
            max_detections: New maximum detections limit.
        """
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            self.logger.info(f"Updated confidence threshold to {confidence_threshold}")

        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            self.logger.info(f"Updated IoU threshold to {iou_threshold}")

        if max_detections is not None:
            self.max_detections = max_detections
            self.logger.info(f"Updated max detections to {max_detections}")

    def benchmark_performance(
        self,
        test_images: List[Union[str, np.ndarray]],
        warmup_runs: int = 5,
        benchmark_runs: int = 20
    ) -> Dict[str, float]:
        """Benchmark detection performance.

        Args:
            test_images: List of test images.
            warmup_runs: Number of warmup runs.
            benchmark_runs: Number of benchmark runs.

        Returns:
            Performance metrics dictionary.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        import time

        self.logger.info("Running performance benchmark...")

        # Warmup
        for _ in range(warmup_runs):
            if len(test_images) > 0:
                self.detect_faces(test_images[0])

        # Benchmark
        times = []
        total_detections = 0

        for i in range(benchmark_runs):
            img_idx = i % len(test_images)

            start_time = time.time()
            detections = self.detect_faces(test_images[img_idx])
            end_time = time.time()

            times.append(end_time - start_time)
            total_detections += len(detections)

        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        avg_detections = total_detections / benchmark_runs

        metrics = {
            'avg_inference_time_ms': avg_time * 1000,
            'std_inference_time_ms': std_time * 1000,
            'fps': fps,
            'avg_detections_per_image': avg_detections,
            'total_runs': benchmark_runs
        }

        self.logger.info(f"Benchmark results: {metrics}")
        return metrics


def create_detector(
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    device: str = 'auto'
) -> MonkeyFaceDetector:
    """Create a monkey face detector instance.

    Args:
        model_path: Path to trained model.
        confidence_threshold: Detection confidence threshold.
        device: Computation device.

    Returns:
        MonkeyFaceDetector instance.
    """
    return MonkeyFaceDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device
    )