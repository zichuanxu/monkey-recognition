"""Image processing utilities."""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union, List
import os
from pathlib import Path

from .logging import get_logger

logger = get_logger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load image from file path.

    Args:
        image_path: Path to image file.

    Returns:
        Image as numpy array in BGR format, or None if loading failed.
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None

        # Try loading with OpenCV first
        image = cv2.imread(image_path)
        if image is not None:
            return image

        # Fallback to PIL
        pil_image = Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert PIL to OpenCV format (RGB to BGR)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image

    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, save_path: str) -> bool:
    """Save image to file.

    Args:
        image: Image as numpy array in BGR format.
        save_path: Path to save the image.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        success = cv2.imwrite(save_path, image)
        if not success:
            logger.error(f"Failed to save image to {save_path}")
        return success

    except Exception as e:
        logger.error(f"Error saving image to {save_path}: {e}")
        return False


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    maintain_aspect_ratio: bool = True,
    fill_color: Tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    """Resize image to target size.

    Args:
        image: Input image as numpy array.
        target_size: Target size as (width, height).
        maintain_aspect_ratio: Whether to maintain aspect ratio.
        fill_color: Fill color for padding when maintaining aspect ratio.

    Returns:
        Resized image.
    """
    if not maintain_aspect_ratio:
        return cv2.resize(image, target_size)

    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h))

    # Create canvas with fill color
    canvas = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)

    # Calculate position to center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    # Place resized image on canvas
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def crop_image(
    image: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    padding: int = 0
) -> Optional[np.ndarray]:
    """Crop image using bounding box coordinates.

    Args:
        image: Input image as numpy array.
        x_min, y_min, x_max, y_max: Bounding box coordinates.
        padding: Additional padding around the crop.

    Returns:
        Cropped image, or None if invalid coordinates.
    """
    h, w = image.shape[:2]

    # Apply padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    # Validate coordinates
    if x_min >= x_max or y_min >= y_max:
        logger.warning(f"Invalid crop coordinates: ({x_min}, {y_min}, {x_max}, {y_max})")
        return None

    return image[y_min:y_max, x_min:x_max]


def normalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                   std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Normalize image using ImageNet statistics.

    Args:
        image: Input image as numpy array (0-255).
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.

    Returns:
        Normalized image.
    """
    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Convert BGR to RGB if needed
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply normalization
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    image = (image - mean) / std

    return image


def denormalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Denormalize image back to [0, 255] range.

    Args:
        image: Normalized image.
        mean: Mean values used for normalization.
        std: Standard deviation values used for normalization.

    Returns:
        Denormalized image in [0, 255] range.
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    # Denormalize
    image = image * std + mean

    # Convert to [0, 255] and uint8
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    # Convert RGB to BGR if needed
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def get_image_files(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> List[str]:
    """Get all image files in a directory.

    Args:
        directory: Directory path to search.
        extensions: Tuple of valid image extensions.

    Returns:
        List of image file paths.
    """
    image_files = []

    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return image_files

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))

    return sorted(image_files)


def validate_image(image_path: str) -> bool:
    """Validate if file is a valid image.

    Args:
        image_path: Path to image file.

    Returns:
        True if valid image, False otherwise.
    """
    try:
        image = load_image(image_path)
        return image is not None and len(image.shape) == 3 and image.shape[2] == 3
    except:
        return False


def get_image_info(image_path: str) -> Optional[dict]:
    """Get basic information about an image.

    Args:
        image_path: Path to image file.

    Returns:
        Dictionary with image information, or None if invalid.
    """
    try:
        image = load_image(image_path)
        if image is None:
            return None

        h, w, c = image.shape
        file_size = os.path.getsize(image_path)

        return {
            'path': image_path,
            'width': w,
            'height': h,
            'channels': c,
            'file_size': file_size,
            'format': Path(image_path).suffix.lower()
        }
    except Exception as e:
        logger.error(f"Error getting image info for {image_path}: {e}")
        return None