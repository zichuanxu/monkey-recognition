"""End-to-end monkey recognition pipeline."""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import time
from pathlib import Path

from ..detection.detector import MonkeyFaceDetector
from ..recognition.recognizer import MonkeyFaceRecognizer
from ..utils.data_structures import BoundingBox, MonkeyDetection
from ..utils.image_utils import load_image, crop_image, save_image
from ..utils.logging import LoggerMixin
from ..utils.validators import InputValidator
from ..utils.exceptions import InferenceError, ErrorCodes
from ..utils.error_handler import handle_errors, safe_execute
from ..utils.visualization import draw_detections, create_detection_summary_image


class MonkeyRecognitionPipeline(LoggerMixin):
    """End-to-end pipeline for monkey face detection and recognition."""

    def __init__(
        self,
        detection_model_path: str,
        recognition_model_path: str,
        database_path: str,
        device: str = 'auto',
        detection_confidence: float = 0.5,
        recognition_confidence: float = 0.6,
        max_faces_per_image: int = 10
    ):
        """Initialize recognition pipeline.

        Args:
            detection_model_path: Path to deten model.
            recognition_model_path: Path to recognition model.
            database_path: Path to feature database.
            device: Device for inference.
            detection_confidence: Detection confidence threshold.
            recognition_confidence: Recognition confidence threshold.
            max_faces_per_image: Maximum faces to process per image.
        """
        self.detection_model_path = detection_model_path
        self.recognition_model_path = recognition_model_path
        self.database_path = database_path
        self.device = device
        self.detection_confidence = InputValidator.validate_confidence_threshold(detection_confidence)
        self.recognition_confidence = InputValidator.validate_confidence_threshold(recognition_confidence)
        self.max_faces_per_image = max_faces_per_image

        # Initialize components
        self.detector = None
        self.recognizer = None

        # Performance tracking
        self.stats = {
            'total_images_processed': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'total_processing_time': 0.0,
            'avg_processing_time_per_image': 0.0
        }

        # Initialize pipeline
        self._initialize_pipeline()

        self.logger.info("MonkeyRecognitionPipeline initialized successfully")

    @handle_errors(InferenceError, reraise=True)
    def _initialize_pipeline(self) -> None:
        """Initialize detection and recognition components."""
        try:
            self.logger.info("Initializing detection component...")
            self.detector = MonkeyFaceDetector(
                model_path=self.detection_model_path,
                confidence_threshold=self.detection_confidence,
                device=self.device
            )

            self.logger.info("Initializing recognition component...")
            self.recognizer = MonkeyFaceRecognizer(
                model_path=self.recognition_model_path,
                database_path=self.database_path,
                device=self.device,
                similarity_threshold=self.recognition_confidence
            )

            # Validate setup
            self._validate_pipeline()

        except Exception as e:
            raise InferenceError(
                f"Failed to initialize pipeline: {str(e)}",
                ErrorCodes.INFERENCE_FAILED
            ) from e

    def _validate_pipeline(self) -> None:
        """Validate pipeline components."""
        # Validate detector
        detector_info = self.detector.get_model_info()
        if not detector_info.get('loaded', False):
            raise InferenceError(
                "Detection model not properly loaded",
                ErrorCodes.MODEL_LOAD_FAILED
            )

        # Validate recognizer
        recognizer_validation = self.recognizer.validate_setup()
        if not recognizer_validation['valid']:
            errors = '; '.join(recognizer_validation['errors'])
            raise InferenceError(
                f"Recognition setup validation failed: {errors}",
                ErrorCodes.RECOGNITION_FAILED
            )

        self.logger.info("Pipeline validation passed")

    @handle_errors(InferenceError, default_return=[])
    def process_image(
        self,
        image: Union[str, np.ndarray],
        return_crops: bool = False,
        return_visualization: bool = False
    ) -> List[MonkeyDetection]:
        """Process single image through complete pipeline.

        Args:
            image: Input image (path or numpy array).
            return_crops: Whether to include cropped face images.
            return_visualization: Whether to include visualization image.

        Returns:
            List of monkey detections with identifications.
        """
        start_time = time.time()

        # Load image if path provided
        if isinstance(image, str):
            image_path = InputValidator.validate_image_path(image)
            img_array = load_image(image_path)
            if img_array is None:
                raise InferenceError(
                    f"Failed to load image: {image_path}",
                    ErrorCodes.DATA_NOT_FOUND
                )
        else:
            img_array = InputValidator.validate_image_array(image)

        try:
            # Step 1: Detect faces
            detections = self.detector.detect_faces(img_array)

            if len(detections) == 0:
                self.logger.debug("No faces detected in image")
                return []

            # Limit number of faces
            if len(detections) > self.max_faces_per_image:
                detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
                detections = detections[:self.max_faces_per_image]
                self.logger.warning(f"Limited to {self.max_faces_per_image} faces per image")

            # Step 2: Crop faces and recognize
            monkey_detections = []

            for detection in detections:
                # Crop face
                face_crop = crop_image(
                    img_array,
                    detection.x_min,
                    detection.y_min,
                    detection.x_max,
                    detection.y_max,
                    padding=10
                )

                if face_crop is None:
                    self.logger.warning("Failed to crop face, skipping")
                    continue

                # Recognize monkey
                monkey_id, recognition_conf = safe_execute(
                    self.recognizer.identify_monkey,
                    face_crop,
                    default_return=("unknown", 0.0)
                )

                # Create monkey detection
                monkey_detection = MonkeyDetection(
                    bbox=detection,
                    monkey_id=monkey_id,
                    recognition_confidence=recognition_conf,
                    detection_confidence=detection.confidence
                )

                # Add crop if requested
                if return_crops:
                    monkey_detection.face_crop = face_crop

                monkey_detections.append(monkey_detection)

            # Add visualization if requested
            if return_visualization:
                vis_image = self._create_visualization(img_array, monkey_detections)
                # Store visualization in first detection for simplicity
                if monkey_detections:
                    monkey_detections[0].visualization = vis_image

            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(1, len(detections), len(monkey_detections), processing_time)

            self.logger.debug(
                f"Processed image: {len(detections)} faces detected, "
                f"{len(monkey_detections)} recognized in {processing_time:.3f}s"
            )

            return monkey_detections

        except Exception as e:
            raise InferenceError(
                f"Image processing failed: {str(e)}",
                ErrorCodes.INFERENCE_FAILED
            ) from e

    def process_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 4,
        show_progress: bool = True
    ) -> List[List[MonkeyDetection]]:
        """Process batch of images.

        Args:
            images: List of images (paths or arrays).
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.

        Returns:
            List of detection lists for each image.
        """
        batch_size = InputValidator.validate_batch_size(batch_size)

        all_results = []

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(images), batch_size), desc="Processing images")
        else:
            iterator = range(0, len(images), batch_size)

        for i in iterator:
            batch = images[i:i + batch_size]
            batch_results = []

            for image in batch:
                result = safe_execute(
                    self.process_image,
                    image,
                    default_return=[]
                )
                batch_results.append(result)

            all_results.extend(batch_results)

        return all_results

    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        save_visualizations: bool = True,
        save_crops: bool = False,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ) -> Dict[str, Any]:
        """Process all images in a directory.

        Args:
            input_dir: Input directory path.
            output_dir: Output directory for results.
            save_visualizations: Whether to save visualization images.
            save_crops: Whether to save face crops.
            image_extensions: Valid image extensions.

        Returns:
            Processing results summary.
        """
        input_dir = InputValidator.validate_directory_path(input_dir)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Find all images
        from ..utils.image_utils import get_image_files
        image_files = get_image_files(input_dir, image_extensions)

        if len(image_files) == 0:
            self.logger.warning(f"No images found in {input_dir}")
            return {'total_images': 0, 'results': []}

        self.logger.info(f"Processing {len(image_files)} images from {input_dir}")

        # Process images
        all_results = []

        for img_path in image_files:
            img_name = Path(img_path).stem

            # Process image
            detections = safe_execute(
                self.process_image,
                img_path,
                return_crops=save_crops,
                return_visualization=save_visualizations,
                default_return=[]
            )

            # Save results if output directory specified
            if output_dir and detections:
                self._save_image_results(
                    img_path,
                    detections,
                    output_dir,
                    save_visualizations,
                    save_crops
                )

            # Store results
            result = {
                'image_path': img_path,
                'image_name': img_name,
                'detections': len(detections),
                'recognized': sum(1 for d in detections if d.monkey_id != 'unknown'),
                'monkey_ids': [d.monkey_id for d in detections]
            }

            all_results.append(result)

        # Create summary
        summary = {
            'total_images': len(image_files),
            'total_detections': sum(r['detections'] for r in all_results),
            'total_recognized': sum(r['recognized'] for r in all_results),
            'unique_monkeys': len(set(
                monkey_id for r in all_results
                for monkey_id in r['monkey_ids']
                if monkey_id != 'unknown'
            )),
            'results': all_results,
            'processing_stats': self.get_statistics()
        }

        # Save summary if output directory specified
        if output_dir:
            import json
            summary_path = os.path.join(output_dir, 'processing_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Directory processing completed: {summary}")
        return summary

    def _save_image_results(
        self,
        image_path: str,
        detections: List[MonkeyDetection],
        output_dir: str,
        save_visualizations: bool,
        save_crops: bool
    ) -> None:
        """Save results for a single image.

        Args:
            image_path: Original image path.
            detections: List of detections.
            output_dir: Output directory.
            save_visualizations: Whether to save visualizations.
            save_crops: Whether to save crops.
        """
        img_name = Path(image_path).stem

        # Save visualization
        if save_visualizations and detections:
            if hasattr(detections[0], 'visualization'):
                vis_image = detections[0].visualization
            else:
                # Create visualization
                original_image = load_image(image_path)
                if original_image is not None:
                    vis_image = self._create_visualization(original_image, detections)
                else:
                    vis_image = None

            if vis_image is not None:
                vis_path = os.path.join(output_dir, f"{img_name}_detections.jpg")
                save_image(vis_image, vis_path)

        # Save crops
        if save_crops:
            crops_dir = os.path.join(output_dir, 'crops', img_name)
            os.makedirs(crops_dir, exist_ok=True)

            for i, detection in enumerate(detections):
                if hasattr(detection, 'face_crop') and detection.face_crop is not None:
                    crop_filename = f"face_{i:02d}_{detection.monkey_id}_{detection.recognition_confidence:.3f}.jpg"
                    crop_path = os.path.join(crops_dir, crop_filename)
                    save_image(detection.face_crop, crop_path)

    def _create_visualization(
        self,
        image: np.ndarray,
        detections: List[MonkeyDetection]
    ) -> np.ndarray:
        """Create visualization image with detections.

        Args:
            image: Original image.
            detections: List of detections.

        Returns:
            Visualization image.
        """
        return create_detection_summary_image(image, detections)

    def _update_stats(
        self,
        images_processed: int,
        faces_detected: int,
        faces_recognized: int,
        processing_time: float
    ) -> None:
        """Update processing statistics.

        Args:
            images_processed: Number of images processed.
            faces_detected: Number of faces detected.
            faces_recognized: Number of faces recognized.
            processing_time: Processing time in seconds.
        """
        self.stats['total_images_processed'] += images_processed
        self.stats['total_faces_detected'] += faces_detected
        self.stats['total_faces_recognized'] += faces_recognized
        self.stats['total_processing_time'] += processing_time

        if self.stats['total_images_processed'] > 0:
            self.stats['avg_processing_time_per_image'] = (
                self.stats['total_processing_time'] / self.stats['total_images_processed']
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.

        Returns:
            Statistics dictionary.
        """
        stats = self.stats.copy()

        # Add derived statistics
        if stats['total_faces_detected'] > 0:
            stats['recognition_rate'] = stats['total_faces_recognized'] / stats['total_faces_detected']
        else:
            stats['recognition_rate'] = 0.0

        if stats['total_processing_time'] > 0:
            stats['fps'] = stats['total_images_processed'] / stats['total_processing_time']
        else:
            stats['fps'] = 0.0

        return stats

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'total_images_processed': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'total_processing_time': 0.0,
            'avg_processing_time_per_image': 0.0
        }

    def update_thresholds(
        self,
        detection_confidence: Optional[float] = None,
        recognition_confidence: Optional[float] = None
    ) -> None:
        """Update confidence thresholds.

        Args:
            detection_confidence: New detection confidence threshold.
            recognition_confidence: New recognition confidence threshold.
        """
        if detection_confidence is not None:
            self.detection_confidence = InputValidator.validate_confidence_threshold(detection_confidence)
            self.detector.update_thresholds(confidence_threshold=detection_confidence)
            self.logger.info(f"Detection confidence updated to {detection_confidence}")

        if recognition_confidence is not None:
            self.recognition_confidence = InputValidator.validate_confidence_threshold(recognition_confidence)
            self.recognizer.update_similarity_threshold(recognition_confidence)
            self.logger.info(f"Recognition confidence updated to {recognition_confidence}")

    def add_monkey_to_database(
        self,
        monkey_id: str,
        images: List[Union[str, np.ndarray]]
    ) -> bool:
        """Add a new monkey to the recognition database.

        Args:
            monkey_id: Monkey identifier.
            images: List of images containing the monkey's face.

        Returns:
            True if successful, False otherwise.
        """
        monkey_id = InputValidator.validate_monkey_id(monkey_id)

        # Extract faces from images
        face_images = []

        for image in images:
            # Process image to detect faces
            detections = safe_execute(
                self.detector.detect_faces,
                image,
                default_return=[]
            )

            if len(detections) == 0:
                self.logger.warning(f"No faces detected in image for monkey {monkey_id}")
                continue

            # Use the face with highest confidence
            best_detection = max(detections, key=lambda x: x.confidence)

            # Load and crop image
            if isinstance(image, str):
                img_array = load_image(image)
            else:
                img_array = image

            if img_array is None:
                continue

            face_crop = crop_image(
                img_array,
                best_detection.x_min,
                best_detection.y_min,
                best_detection.x_max,
                best_detection.y_max,
                padding=10
            )

            if face_crop is not None:
                face_images.append(face_crop)

        if len(face_images) == 0:
            self.logger.error(f"No valid face images found for monkey {monkey_id}")
            return False

        # Add to database
        return self.recognizer.add_monkey_to_database(monkey_id, face_images)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information.

        Returns:
            Pipeline information dictionary.
        """
        info = {
            'detection_model': self.detection_model_path,
            'recognition_model': self.recognition_model_path,
            'database_path': self.database_path,
            'device': self.device,
            'detection_confidence': self.detection_confidence,
            'recognition_confidence': self.recognition_confidence,
            'max_faces_per_image': self.max_faces_per_image,
            'statistics': self.get_statistics()
        }

        # Add component info
        if self.detector:
            info['detector_info'] = self.detector.get_model_info()

        if self.recognizer:
            info['recognizer_info'] = self.recognizer.get_model_info()
            info['database_stats'] = self.recognizer.get_database_statistics()

        return info

    def benchmark_pipeline(
        self,
        test_images: List[Union[str, np.ndarray]],
        warmup_runs: int = 3,
        benchmark_runs: int = 10
    ) -> Dict[str, Any]:
        """Benchmark pipeline performance.

        Args:
            test_images: Test images for benchmarking.
            warmup_runs: Number of warmup runs.
            benchmark_runs: Number of benchmark runs.

        Returns:
            Benchmark results dictionary.
        """
        if len(test_images) == 0:
            return {'error': 'No test images provided'}

        self.logger.info("Running pipeline benchmark...")

        # Reset statistics
        self.reset_statistics()

        # Warmup
        for _ in range(warmup_runs):
            self.process_image(test_images[0])

        # Reset statistics after warmup
        self.reset_statistics()

        # Benchmark
        start_time = time.time()

        for i in range(benchmark_runs):
            img_idx = i % len(test_images)
            self.process_image(test_images[img_idx])

        total_time = time.time() - start_time

        # Get final statistics
        stats = self.get_statistics()

        benchmark_results = {
            'total_benchmark_time': total_time,
            'images_processed': stats['total_images_processed'],
            'faces_detected': stats['total_faces_detected'],
            'faces_recognized': stats['total_faces_recognized'],
            'avg_time_per_image': stats['avg_processing_time_per_image'],
            'fps': stats['fps'],
            'recognition_rate': stats['recognition_rate']
        }

        self.logger.info(f"Benchmark completed: {benchmark_results}")
        return benchmark_results


def create_pipeline(
    detection_model_path: str,
    recognition_model_path: str,
    database_path: str,
    device: str = 'auto',
    detection_confidence: float = 0.5,
    recognition_confidence: float = 0.6
) -> MonkeyRecognitionPipeline:
    """Create a monkey recognition pipeline instance.

    Args:
        detection_model_path: Path to detection model.
        recognition_model_path: Path to recognition model.
        database_path: Path to feature database.
        device: Device for inference.
        detection_confidence: Detection confidence threshold.
        recognition_confidence: Recognition confidence threshold.

    Returns:
        MonkeyRecognitionPipeline instance.
    """
    return MonkeyRecognitionPipeline(
        detection_model_path=detection_model_path,
        recognition_model_path=recognition_model_path,
        database_path=database_path,
        device=device,
        detection_confidence=detection_confidence,
        recognition_confidence=recognition_confidence
    )