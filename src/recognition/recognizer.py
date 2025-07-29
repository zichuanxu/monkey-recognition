"""Monkey face recognition component."""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import os
from pathlib import Path

from .model import MonkeyRecognitionModel
from .database import FeatureDatabase
from ..utils.data_structures import MonkeyDetection, BoundingBox
from ..utils.image_utils import load_image, crop_image, resize_image
from ..utils.logging import LoggerMixin
from ..utils.validators import InputValidator
from ..utils.exceptions import RecognitionError, ModelError, DatabaseError, ErrorCodes
from ..utils.error_handler import handle_errors, safe_execute
from ..data.preprocessor import DataPreprocessor


class MonkeyFaceRecognizer(LoggerMixin):
    """Main component for monkey face recognition."""

    def __init__(
        self,
        model_path: str,
        database_path: str,
        device: str = 'auto',
        similarity_threshold: float = 0.6,
        embedding_size: int = 512
    ):
        """Initialize monkey face recognizer.

        Args:
            model_path: Path to trained recognition model.
            database_path: Path to feature database.
            device: Device for inference ('cpu', 'cuda', 'auto').
            similarity_threshold: Threshold for recognition confidence.
            embedding_size: Expected embedding dimension.
        """
        self.model_path = InputValidator.validate_model_path(model_path)
        self.database_path = database_path
        self.device = self._setup_device(device)
        self.similarity_threshold = InputValidator.validate_confidence_threshold(similarity_threshold)
        self.embedding_size = InputValidator.validate_embedding_size(embedding_size)

        # Initialize components
        self.model = None
        self.database = None
        self.preprocessor = DataPreprocessor(recognition_image_size=(224, 224))

        # Load model and database
        self._load_model()
        self._load_database()

        self.logger.info(
            f"MonkeyFaceRecognizer initialized: model={model_path}, "
            f"database={database_path}, device={self.device}"
        )

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device.

        Args:
            device: Device specification.

        Returns:
            PyTorch device.
        """
        device = InputValidator.validate_device(device)

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        return torch.device(device)

    @handle_errors(ModelError, reraise=True)
    def _load_model(self) -> None:
        """Load recognition model from checkpoint."""
        try:
            self.logger.info(f"Loading recognition model from {self.model_path}")

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Extract model configuration
            config = checkpoint.get('config', {})
            num_classes = checkpoint.get('num_classes', 1)

            # Create model
            from .model import create_recognition_model

            self.model = create_recognition_model(
                num_classes=num_classes,
                backbone=config.get('backbone', 'resnet50'),
                embedding_size=config.get('embedding_size', self.embedding_size),
                margin_loss='arcface',
                pretrained=False  # We're loading trained weights
            )

            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            # Store class names if available
            self.class_names = checkpoint.get('class_names', [])

            self.logger.info("Recognition model loaded successfully")

        except Exception as e:
            raise ModelError(
                f"Failed to load recognition model: {str(e)}",
                ErrorCodes.MODEL_LOAD_FAILED
            ) from e

    @handle_errors(DatabaseError, reraise=True)
    def _load_database(self) -> None:
        """Load feature database."""
        try:
            self.logger.info(f"Loading feature database from {self.database_path}")

            if not os.path.exists(self.database_path):
                self.logger.warning(f"Database file not found: {self.database_path}")
                self.logger.info("Creating empty database")

                from .database import create_feature_database
                self.database = create_feature_database(
                    embedding_size=self.embedding_size,
                    similarity_metric='cosine',
                    use_faiss=True
                )
            else:
                from .database import FeatureDatabase
                self.database = FeatureDatabase(embedding_size=self.embedding_size)
                self.database.load(self.database_path)

            db_stats = self.database.get_statistics()
            self.logger.info(f"Database loaded: {db_stats['num_monkeys']} monkeys, {db_stats['total_features']} features")

        except Exception as e:
            raise DatabaseError(
                f"Failed to load feature database: {str(e)}",
                ErrorCodes.DATABASE_CONNECTION_FAILED
            ) from e

    @handle_errors(RecognitionError, default_return=np.array([]))
    def extract_features(self, face_image: np.ndarray) -> np.ndarray:
        """Extract feature embeddings from face image.

        Args:
            face_image: Face image as numpy array.

        Returns:
            Feature embedding vector.
        """
        if self.model is None:
            raise RecognitionError(
                "Model not loaded",
                ErrorCodes.MODEL_NOT_FOUND
            )

        # Validate input
        face_image = InputValidator.validate_image_array(face_image)

        try:
            # Preprocess image
            processed_image = self.preprocessor.preprocess_for_recognition(
                face_image, training=False, return_tensor=True
            )

            # Add batch dimension
            if processed_image.dim() == 3:
                processed_image = processed_image.unsqueeze(0)

            # Move to device
            processed_image = processed_image.to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model.extract_features(processed_image)
                features_np = features.cpu().numpy()

            return features_np.flatten()

        except Exception as e:
            raise RecognitionError(
                f"Feature extraction failed: {str(e)}",
                ErrorCodes.FEATURE_EXTRACTION_FAILED
            ) from e

    @handle_errors(RecognitionError, default_return=("unknown", 0.0))
    def identify_monkey(
        self,
        face_image: np.ndarray,
        top_k: int = 1
    ) -> Union[Tuple[str, float], List[Tuple[str, float, Dict[str, Any]]]]:
        """Identify monkey from face image.

        Args:
            face_image: Face image as numpy array.
            top_k: Number of top matches to return.

        Returns:
            If top_k=1: Tuple of (monkey_id, confidence)
            If top_k>1: List of (monkey_id, confidence, metadata) tuples
        """
        if self.database is None:
            raise RecognitionError(
                "Database not loaded",
                ErrorCodes.DATABASE_CONNECTION_FAILED
            )

        # Extract features
        features = self.extract_features(face_image)

        if features.size == 0:
            if top_k == 1:
                return "unknown", 0.0
            else:
                return []

        try:
            # Find closest matches
            matches = self.database.find_closest_match(
                features,
                top_k=top_k,
                threshold=self.similarity_threshold
            )

            if len(matches) == 0:
                if top_k == 1:
                    return "unknown", 0.0
                else:
                    return []

            if top_k == 1:
                monkey_id, confidence, _ = matches[0]
                return monkey_id, confidence
            else:
                return matches

        except Exception as e:
            raise RecognitionError(
                f"Monkey identification failed: {str(e)}",
                ErrorCodes.RECOGNITION_FAILED
            ) from e

    def recognize_faces(
        self,
        face_images: List[np.ndarray],
        batch_size: int = 8
    ) -> List[Tuple[str, float]]:
        """Recognize multiple faces in batch.

        Args:
            face_images: List of face images.
            batch_size: Batch size for processing.

        Returns:
            List of (monkey_id, confidence) tuples.
        """
        batch_size = InputValidator.validate_batch_size(batch_size)

        results = []

        # Process in batches
        for i in range(0, len(face_images), batch_size):
            batch = face_images[i:i + batch_size]

            batch_results = []
            for face_image in batch:
                result = safe_execute(
                    self.identify_monkey,
                    face_image,
                    default_return=("unknown", 0.0)
                )
                batch_results.append(result)

            results.extend(batch_results)

        return results

    def add_monkey_to_database(
        self,
        monkey_id: str,
        face_images: List[np.ndarray],
        save_database: bool = True
    ) -> bool:
        """Add a new monkey to the database.

        Args:
            monkey_id: Monkey identifier.
            face_images: List of face images for this monkey.
            save_database: Whether to save database after adding.

        Returns:
            True if successful, False otherwise.
        """
        monkey_id = InputValidator.validate_monkey_id(monkey_id)

        if self.database is None:
            self.logger.error("Database not loaded")
            return False

        try:
            # Extract features for all images
            all_features = []

            for face_image in face_images:
                features = self.extract_features(face_image)
                if features.size > 0:
                    all_features.append(features)

            if len(all_features) == 0:
                self.logger.warning(f"No valid features extracted for monkey {monkey_id}")
                return False

            # Add to database
            features_array = np.vstack(all_features)
            self.database.add_monkey(monkey_id, features_array)

            # Save database if requested
            if save_database:
                self.save_database()

            self.logger.info(f"Added monkey {monkey_id} with {len(all_features)} features")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add monkey {monkey_id}: {e}")
            return False

    def remove_monkey_from_database(
        self,
        monkey_id: str,
        save_database: bool = True
    ) -> bool:
        """Remove a monkey from the database.

        Args:
            monkey_id: Monkey identifier to remove.
            save_database: Whether to save database after removal.

        Returns:
            True if successful, False otherwise.
        """
        monkey_id = InputValidator.validate_monkey_id(monkey_id)

        if self.database is None:
            self.logger.error("Database not loaded")
            return False

        try:
            success = self.database.remove_monkey(monkey_id)

            if success and save_database:
                self.save_database()

            return success

        except Exception as e:
            self.logger.error(f"Failed to remove monkey {monkey_id}: {e}")
            return False

    def save_database(self) -> bool:
        """Save feature database to file.

        Returns:
            True if successful, False otherwise.
        """
        if self.database is None:
            self.logger.error("Database not loaded")
            return False

        try:
            self.database.save(self.database_path)
            self.logger.info(f"Database saved to {self.database_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save database: {e}")
            return False

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Statistics dictionary.
        """
        if self.database is None:
            return {'error': 'Database not loaded'}

        return self.database.get_statistics()

    def get_monkey_list(self) -> List[str]:
        """Get list of monkeys in database.

        Returns:
            List of monkey IDs.
        """
        if self.database is None:
            return []

        return self.database.monkey_ids.copy()

    def update_similarity_threshold(self, threshold: float) -> None:
        """Update similarity threshold for recognition.

        Args:
            threshold: New similarity threshold.
        """
        self.similarity_threshold = InputValidator.validate_confidence_threshold(threshold)
        self.logger.info(f"Similarity threshold updated to {threshold}")

    def benchmark_recognition_speed(
        self,
        test_images: List[np.ndarray],
        warmup_runs: int = 5,
        benchmark_runs: int = 20
    ) -> Dict[str, float]:
        """Benchmark recognition speed.

        Args:
            test_images: Test images for benchmarking.
            warmup_runs: Number of warmup runs.
            benchmark_runs: Number of benchmark runs.

        Returns:
            Performance metrics dictionary.
        """
        if len(test_images) == 0:
            return {'error': 'No test images provided'}

        import time

        self.logger.info("Running recognition speed benchmark...")

        # Warmup
        for _ in range(warmup_runs):
            self.identify_monkey(test_images[0])

        # Benchmark
        times = []

        for i in range(benchmark_runs):
            img_idx = i % len(test_images)

            start_time = time.time()
            self.identify_monkey(test_images[img_idx])
            end_time = time.time()

            times.append(end_time - start_time)

        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        metrics = {
            'avg_recognition_time_ms': avg_time * 1000,
            'std_recognition_time_ms': std_time * 1000,
            'fps': fps,
            'total_runs': benchmark_runs
        }

        self.logger.info(f"Benchmark results: {metrics}")
        return metrics

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Model information dictionary.
        """
        if self.model is None:
            return {'error': 'Model not loaded'}

        info = self.model.get_model_info()
        info.update({
            'model_path': self.model_path,
            'device': str(self.device),
            'similarity_threshold': self.similarity_threshold,
            'embedding_size': self.embedding_size
        })

        return info

    def validate_setup(self) -> Dict[str, Any]:
        """Validate recognizer setup.

        Returns:
            Validation results dictionary.
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        # Check model
        if self.model is None:
            results['valid'] = False
            results['errors'].append("Model not loaded")
        else:
            results['info']['model_loaded'] = True
            results['info']['model_device'] = str(self.device)

        # Check database
        if self.database is None:
            results['valid'] = False
            results['errors'].append("Database not loaded")
        else:
            db_stats = self.database.get_statistics()
            results['info']['database_loaded'] = True
            results['info']['num_monkeys'] = db_stats['num_monkeys']
            results['info']['total_features'] = db_stats['total_features']

            if db_stats['num_monkeys'] == 0:
                results['warnings'].append("Database is empty")

        # Check device availability
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            results['valid'] = False
            results['errors'].append("CUDA device specified but not available")

        return results


def create_recognizer(
    model_path: str,
    database_path: str,
    device: str = 'auto',
    similarity_threshold: float = 0.6,
    embedding_size: int = 512
) -> MonkeyFaceRecognizer:
    """Create a monkey face recognizer instance.

    Args:
        model_path: Path to trained recognition model.
        database_path: Path to feature database.
        device: Device for inference.
        similarity_threshold: Recognition confidence threshold.
        embedding_size: Expected embedding dimension.

    Returns:
        MonkeyFaceRecognizer instance.
    """
    return MonkeyFaceRecognizer(
        model_path=model_path,
        database_path=database_path,
        device=device,
        similarity_threshold=similarity_threshold,
        embedding_size=embedding_size
    )