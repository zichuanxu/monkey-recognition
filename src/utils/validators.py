"""Input validation utilities for the monkey recognition system."""

import os
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import cv2

from .exceptions import ValidationError, ErrorCodes
from .logging import LoggerMixin


class InputValidator(LoggerMixin):
    """Comprehensive input validation for the monkey recognition system."""

    @staticmethod
    def validate_image_path(image_path: str) -> str:
        """Validate image file path.

        Args:
            image_path: Path to image file.

        Returns:
            Validated image path.

        Raises:
            ValidationError: If path is invalid.
        """
        if not isinstance(image_path, (str, Path)):
            raise ValidationError(
                f"Image path must be string or Path, got {type(image_path)}",
                ErrorCodes.INPUT_INVALID
            )

        image_path = str(image_path)

        if not image_path.strip():
            raise ValidationError(
                "Image path cannot be empty",
                ErrorCodes.INPUT_INVALID
            )

        if not os.path.exists(image_path):
            raise ValidationError(
                f"Image file not found: {image_path}",
                ErrorCodes.FILE_NOT_FOUND
            )

        if not os.path.isfile(image_path):
            raise ValidationError(
                f"Path is not a file: {image_path}",
                ErrorCodes.INPUT_INVALID
            )

        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        ext = Path(image_path).suffix.lower()

        if ext not in valid_extensions:
            raise ValidationError(
                f"Unsupported image format: {ext}. Supported: {valid_extensions}",
                ErrorCodes.DATA_FORMAT_INVALID
            )

        return image_path

    @staticmethod
    def validate_image_array(image: np.ndarray) -> np.ndarray:
        """Validate image numpy array.

        Args:
            image: Image array.

        Returns:
            Validated image array.

        Raises:
            ValidationError: If image array is invalid.
        """
        if not isinstance(image, np.ndarray):
            raise ValidationError(
                f"Image must be numpy array, got {type(image)}",
                ErrorCodes.INPUT_INVALID
            )

        if image.size == 0:
            raise ValidationError(
                "Image array is empty",
                ErrorCodes.INPUT_INVALID
            )

        if len(image.shape) not in [2, 3]:
            raise ValidationError(
                f"Image must be 2D or 3D array, got shape {image.shape}",
                ErrorCodes.INPUT_INVALID
            )

        if len(image.shape) == 3:
            if image.shape[2] not in [1, 3, 4]:
                raise ValidationError(
                    f"Image must have 1, 3, or 4 channels, got {image.shape[2]}",
                    ErrorCodes.INPUT_INVALID
                )

        # Check data type
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            raise ValidationError(
                f"Unsupported image dtype: {image.dtype}",
                ErrorCodes.DATA_FORMAT_INVALID
            )

        # Check value range
        if image.dtype == np.uint8:
            if image.min() < 0 or image.max() > 255:
                raise ValidationError(
                    f"uint8 image values must be in [0, 255], got [{image.min()}, {image.max()}]",
                    ErrorCodes.INPUT_INVALID
                )
        elif image.dtype in [np.float32, np.float64]:
            if image.min() < -1.0 or image.max() > 1.0:
                # Allow both [0, 1] and [-1, 1] ranges for float images
                if not (0.0 <= image.min() and image.max() <= 1.0):
                    raise ValidationError(
                        f"Float image values must be in [0, 1] or [-1, 1], got [{image.min():.3f}, {image.max():.3f}]",
                        ErrorCodes.INPUT_INVALID
                    )

        return image

    @staticmethod
    def validate_directory_path(dir_path: str, must_exist: bool = True) -> str:
        """Validate directory path.

        Args:
            dir_path: Directory path.
            must_exist: Whether directory must exist.

        Returns:
            Validated directory path.

        Raises:
            ValidationError: If directory path is invalid.
        """
        if not isinstance(dir_path, (str, Path)):
            raise ValidationError(
                f"Directory path must be string or Path, got {type(dir_path)}",
                ErrorCodes.INPUT_INVALID
            )

        dir_path = str(dir_path)

        if not dir_path.strip():
            raise ValidationError(
                "Directory path cannot be empty",
                ErrorCodes.INPUT_INVALID
            )

        if must_exist:
            if not os.path.exists(dir_path):
                raise ValidationError(
                    f"Directory not found: {dir_path}",
                    ErrorCodes.FILE_NOT_FOUND
                )

            if not os.path.isdir(dir_path):
                raise ValidationError(
                    f"Path is not a directory: {dir_path}",
                    ErrorCodes.INPUT_INVALID
                )

        return dir_path

    @staticmethod
    def validate_model_path(model_path: str) -> str:
        """Validate model file path.

        Args:
            model_path: Path to model file.

        Returns:
            Validated model path.

        Raises:
            ValidationError: If model path is invalid.
        """
        if not isinstance(model_path, (str, Path)):
            raise ValidationError(
                f"Model path must be string or Path, got {type(model_path)}",
                ErrorCodes.INPUT_INVALID
            )

        model_path = str(model_path)

        if not model_path.strip():
            raise ValidationError(
                "Model path cannot be empty",
                ErrorCodes.INPUT_INVALID
            )

        if not os.path.exists(model_path):
            raise ValidationError(
                f"Model file not found: {model_path}",
                ErrorCodes.MODEL_NOT_FOUND
            )

        if not os.path.isfile(model_path):
            raise ValidationError(
                f"Model path is not a file: {model_path}",
                ErrorCodes.INPUT_INVALID
            )

        # Check file extension
        valid_extensions = {'.pt', '.pth', '.onnx', '.pb', '.tflite'}
        ext = Path(model_path).suffix.lower()

        if ext not in valid_extensions:
            raise ValidationError(
                f"Unsupported model format: {ext}. Supported: {valid_extensions}",
                ErrorCodes.MODEL_INCOMPATIBLE
            )

        return model_path

    @staticmethod
    def validate_confidence_threshold(threshold: float) -> float:
        """Validate confidence threshold.

        Args:
            threshold: Confidence threshold value.

        Returns:
            Validated threshold.

        Raises:
            ValidationError: If threshold is invalid.
        """
        if not isinstance(threshold, (int, float)):
            raise ValidationError(
                f"Confidence threshold must be numeric, got {type(threshold)}",
                ErrorCodes.INPUT_INVALID
            )

        threshold = float(threshold)

        if not (0.0 <= threshold <= 1.0):
            raise ValidationError(
                f"Confidence threshold must be in [0, 1], got {threshold}",
                ErrorCodes.PARAMETER_INVALID
            )

        return threshold

    @staticmethod
    def validate_iou_threshold(threshold: float) -> float:
        """Validate IoU threshold.

        Args:
            threshold: IoU threshold value.

        Returns:
            Validated threshold.

        Raises:
            ValidationError: If threshold is invalid.
        """
        if not isinstance(threshold, (int, float)):
            raise ValidationError(
                f"IoU threshold must be numeric, got {type(threshold)}",
                ErrorCodes.INPUT_INVALID
            )

        threshold = float(threshold)

        if not (0.0 <= threshold <= 1.0):
            raise ValidationError(
                f"IoU threshold must be in [0, 1], got {threshold}",
                ErrorCodes.PARAMETER_INVALID
            )

        return threshold

    @staticmethod
    def validate_batch_size(batch_size: int, max_batch_size: int = 128) -> int:
        """Validate batch size.

        Args:
            batch_size: Batch size value.
            max_batch_size: Maximum allowed batch size.

        Returns:
            Validated batch size.

        Raises:
            ValidationError: If batch size is invalid.
        """
        if not isinstance(batch_size, int):
            raise ValidationError(
                f"Batch size must be integer, got {type(batch_size)}",
                ErrorCodes.INPUT_INVALID
            )

        if batch_size <= 0:
            raise ValidationError(
                f"Batch size must be positive, got {batch_size}",
                ErrorCodes.PARAMETER_INVALID
            )

        if batch_size > max_batch_size:
            raise ValidationError(
                f"Batch size too large: {batch_size} > {max_batch_size}",
                ErrorCodes.BATCH_SIZE_EXCEEDED
            )

        return batch_size

    @staticmethod
    def validate_image_size(size: Tuple[int, int]) -> Tuple[int, int]:
        """Validate image size tuple.

        Args:
            size: Image size as (width, height).

        Returns:
            Validated image size.

        Raises:
            ValidationError: If size is invalid.
        """
        if not isinstance(size, (tuple, list)):
            raise ValidationError(
                f"Image size must be tuple or list, got {type(size)}",
                ErrorCodes.INPUT_INVALID
            )

        if len(size) != 2:
            raise ValidationError(
                f"Image size must have 2 elements, got {len(size)}",
                ErrorCodes.INPUT_INVALID
            )

        width, height = size

        if not isinstance(width, int) or not isinstance(height, int):
            raise ValidationError(
                f"Image size elements must be integers, got {type(width)}, {type(height)}",
                ErrorCodes.INPUT_INVALID
            )

        if width <= 0 or height <= 0:
            raise ValidationError(
                f"Image size must be positive, got ({width}, {height})",
                ErrorCodes.PARAMETER_INVALID
            )

        if width > 4096 or height > 4096:
            raise ValidationError(
                f"Image size too large: ({width}, {height}), max: (4096, 4096)",
                ErrorCodes.PARAMETER_INVALID
            )

        return (width, height)

    @staticmethod
    def validate_device(device: str) -> str:
        """Validate device specification.

        Args:
            device: Device specification ('cpu', 'cuda', 'auto').

        Returns:
            Validated device.

        Raises:
            ValidationError: If device is invalid.
        """
        if not isinstance(device, str):
            raise ValidationError(
                f"Device must be string, got {type(device)}",
                ErrorCodes.INPUT_INVALID
            )

        device = device.lower().strip()

        valid_devices = {'cpu', 'cuda', 'auto'}
        if device not in valid_devices:
            raise ValidationError(
                f"Invalid device: {device}. Valid options: {valid_devices}",
                ErrorCodes.PARAMETER_INVALID
            )

        # Check CUDA availability if requested
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValidationError(
                "CUDA device requested but not available",
                ErrorCodes.GPU_NOT_AVAILABLE
            )

        return device

    @staticmethod
    def validate_embedding_size(size: int) -> int:
        """Validate embedding size.

        Args:
            size: Embedding dimension size.

        Returns:
            Validated embedding size.

        Raises:
            ValidationError: If size is invalid.
        """
        if not isinstance(size, int):
            raise ValidationError(
                f"Embedding size must be integer, got {type(size)}",
                ErrorCodes.INPUT_INVALID
            )

        if size <= 0:
            raise ValidationError(
                f"Embedding size must be positive, got {size}",
                ErrorCodes.PARAMETER_INVALID
            )

        if size > 4096:
            raise ValidationError(
                f"Embedding size too large: {size}, max: 4096",
                ErrorCodes.PARAMETER_INVALID
            )

        # Check if power of 2 (recommended for efficiency)
        if size & (size - 1) != 0:
            InputValidator().logger.warning(
                f"Embedding size {size} is not a power of 2, which may affect performance"
            )

        return size

    @staticmethod
    def validate_monkey_id(monkey_id: str) -> str:
        """Validate monkey identifier.

        Args:
            monkey_id: Monkey identifier string.

        Returns:
            Validated monkey ID.

        Raises:
            ValidationError: If monkey ID is invalid.
        """
        if not isinstance(monkey_id, str):
            raise ValidationError(
                f"Monkey ID must be string, got {type(monkey_id)}",
                ErrorCodes.INPUT_INVALID
            )

        monkey_id = monkey_id.strip()

        if not monkey_id:
            raise ValidationError(
                "Monkey ID cannot be empty",
                ErrorCodes.INPUT_INVALID
            )

        if len(monkey_id) > 100:
            raise ValidationError(
                f"Monkey ID too long: {len(monkey_id)} characters, max: 100",
                ErrorCodes.PARAMETER_INVALID
            )

        # Check for invalid characters
        invalid_chars = {'/', '\\', ':', '*', '?', '"', '<', '>', '|'}
        if any(char in monkey_id for char in invalid_chars):
            raise ValidationError(
                f"Monkey ID contains invalid characters: {invalid_chars}",
                ErrorCodes.INPUT_INVALID
            )

        return monkey_id

    @staticmethod
    def validate_feature_vector(features: np.ndarray, expected_size: Optional[int] = None) -> np.ndarray:
        """Validate feature vector.

        Args:
            features: Feature vector array.
            expected_size: Expected feature dimension.

        Returns:
            Validated feature vector.

        Raises:
            ValidationError: If features are invalid.
        """
        if not isinstance(features, np.ndarray):
            raise ValidationError(
                f"Features must be numpy array, got {type(features)}",
                ErrorCodes.INPUT_INVALID
            )

        if features.size == 0:
            raise ValidationError(
                "Feature vector is empty",
                ErrorCodes.INPUT_INVALID
            )

        if len(features.shape) > 2:
            raise ValidationError(
                f"Features must be 1D or 2D array, got shape {features.shape}",
                ErrorCodes.INPUT_INVALID
            )

        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        if expected_size is not None:
            if features.shape[1] != expected_size:
                raise ValidationError(
                    f"Feature dimension mismatch: expected {expected_size}, got {features.shape[1]}",
                    ErrorCodes.INPUT_INVALID
                )

        # Check for NaN or infinite values
        if not np.isfinite(features).all():
            raise ValidationError(
                "Feature vector contains NaN or infinite values",
                ErrorCodes.INPUT_INVALID
            )

        return features

    @staticmethod
    def validate_config_dict(config: Dict[str, Any], required_keys: List[str]) -> Dict[str, Any]:
        """Validate configuration dictionary.

        Args:
            config: Configuration dictionary.
            required_keys: List of required keys.

        Returns:
            Validated configuration.

        Raises:
            ValidationError: If configuration is invalid.
        """
        if not isinstance(config, dict):
            raise ValidationError(
                f"Configuration must be dictionary, got {type(config)}",
                ErrorCodes.INPUT_INVALID
            )

        # Check required keys
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValidationError(
                f"Missing required configuration keys: {missing_keys}",
                ErrorCodes.CONFIG_MISSING_REQUIRED
            )

        return config

    @staticmethod
    def validate_file_permissions(file_path: str, read: bool = True, write: bool = False) -> str:
        """Validate file permissions.

        Args:
            file_path: Path to file.
            read: Whether read permission is required.
            write: Whether write permission is required.

        Returns:
            Validated file path.

        Raises:
            ValidationError: If permissions are insufficient.
        """
        file_path = str(file_path)

        if not os.path.exists(file_path):
            raise ValidationError(
                f"File not found: {file_path}",
                ErrorCodes.FILE_NOT_FOUND
            )

        if read and not os.access(file_path, os.R_OK):
            raise ValidationError(
                f"No read permission for file: {file_path}",
                ErrorCodes.FILE_PERMISSION_DENIED
            )

        if write and not os.access(file_path, os.W_OK):
            raise ValidationError(
                f"No write permission for file: {file_path}",
                ErrorCodes.FILE_PERMISSION_DENIED
            )

        return file_path

    @staticmethod
    def validate_memory_requirements(required_mb: float, safety_factor: float = 1.5) -> None:
        """Validate memory requirements.

        Args:
            required_mb: Required memory in MB.
            safety_factor: Safety factor for memory estimation.

        Raises:
            ValidationError: If insufficient memory.
        """
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / (1024 * 1024)

            required_with_safety = required_mb * safety_factor

            if available_mb < required_with_safety:
                raise ValidationError(
                    f"Insufficient memory: required {required_with_safety:.1f} MB, "
                    f"available {available_mb:.1f} MB",
                    ErrorCodes.MEMORY_INSUFFICIENT
                )
        except ImportError:
            # psutil not available, skip memory check
            pass

    @staticmethod
    def validate_gpu_memory(required_mb: float) -> None:
        """Validate GPU memory requirements.

        Args:
            required_mb: Required GPU memory in MB.

        Raises:
            ValidationError: If insufficient GPU memory.
        """
        if not torch.cuda.is_available():
            return

        try:
            available_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            used_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
            free_mb = available_mb - used_mb

            if free_mb < required_mb:
                raise ValidationError(
                    f"Insufficient GPU memory: required {required_mb:.1f} MB, "
                    f"available {free_mb:.1f} MB",
                    ErrorCodes.MEMORY_INSUFFICIENT
                )
        except Exception:
            # Skip GPU memory check if unable to determine
            pass