"""Data preprocessing utilities."""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, List, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..utils.image_utils import resize_image, normalize_image, denormalize_image
from ..utils.logging import LoggerMixin


class DataPreprocessor(LoggerMixin):
    """Data preprocessor for detection and recognition tasks."""

    def __init__(
        self,
        detection_image_size: Tuple[int, int] = (640, 640),
        recognition_image_size: Tuple[int, int] = (224, 224)
    ):
        """Initialize data preprocessor.

        Args:
            detection_image_size: Target size for detection images.
            recognition_image_size: Target size for recognition images.
        """
        self.detection_image_size = detection_image_size
        self.recognition_image_size = recognition_image_size

        # Initialize transforms
        self.detection_transform = self._get_detection_transform()
        self.recognition_transform = self._get_recognition_transform()
        self.recognition_train_transform = self._get_recognition_train_transform()

    def _get_detection_transform(self) -> A.Compose:
        """Get transform for detection preprocessing.

        Returns:
            Albumentations transform for detection.
        """
        return A.Compose([
            A.Resize(
                height=self.detection_image_size[1],
                width=self.detection_image_size[0]
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # Keep in [0, 1] range
            ToTensorV2()
        ])

    def _get_recognition_transform(self) -> A.Compose:
        """Get transform for recognition preprocessing (inference).

        Returns:
            Albumentations transform for recognition.
        """
        return A.Compose([
            A.Resize(
                height=self.recognition_image_size[1],
                width=self.recognition_image_size[0]
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def _get_recognition_train_transform(self) -> A.Compose:
        """Get transform for recognition training with augmentation.

        Returns:
            Albumentations transform for recognition training.
        """
        return A.Compose([
            A.Resize(
                height=self.recognition_image_size[1],
                width=self.recognition_image_size[0]
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def preprocess_for_detection(
        self,
        image: np.ndarray,
        return_tensor: bool = True
    ) -> np.ndarray:
        """Preprocess image for detection model.

        Args:
            image: Input image in BGR format.
            return_tensor: Whether to return as tensor.

        Returns:
            Preprocessed image.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if return_tensor:
            # Apply transform
            transformed = self.detection_transform(image=image_rgb)
            return transformed['image']
        else:
            # Manual preprocessing
            resized = resize_image(
                image,
                self.detection_image_size,
                maintain_aspect_ratio=True
            )
            return resized

    def preprocess_for_recognition(
        self,
        image: np.ndarray,
        training: bool = False,
        return_tensor: bool = True
    ) -> torch.Tensor:
        """Preprocess image for recognition model.

        Args:
            image: Input image in BGR format.
            training: Whether this is for training (applies augmentation).
            return_tensor: Whether to return as tensor.

        Returns:
            Preprocessed image tensor.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Choose transform based on training mode
        if training:
            transform = self.recognition_train_transform
        else:
            transform = self.recognition_transform

        if return_tensor:
            # Apply transform
            transformed = transform(image=image_rgb)
            return transformed['image']
        else:
            # Manual preprocessing
            resized = resize_image(
                image,
                self.recognition_image_size,
                maintain_aspect_ratio=False
            )
            normalized = normalize_image(resized)
            return normalized

    def preprocess_batch_for_detection(
        self,
        images: List[np.ndarray]
    ) -> torch.Tensor:
        """Preprocess batch of images for detection.

        Args:
            images: List of input images in BGR format.

        Returns:
            Batch tensor of preprocessed images.
        """
        batch_tensors = []

        for image in images:
            tensor = self.preprocess_for_detection(image, return_tensor=True)
            batch_tensors.append(tensor)

        return torch.stack(batch_tensors)

    def preprocess_batch_for_recognition(
        self,
        images: List[np.ndarray],
        training: bool = False
    ) -> torch.Tensor:
        """Preprocess batch of images for recognition.

        Args:
            images: List of input images in BGR format.
            training: Whether this is for training.

        Returns:
            Batch tensor of preprocessed images.
        """
        batch_tensors = []

        for image in images:
            tensor = self.preprocess_for_recognition(
                image, training=training, return_tensor=True
            )
            batch_tensors.append(tensor)

        return torch.stack(batch_tensors)

    def postprocess_detection_output(
        self,
        detections: List[Dict[str, Any]],
        original_image_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Postprocess detection model output.

        Args:
            detections: Raw detection output from model.
            original_image_size: Original image size (width, height).

        Returns:
            Postprocessed detections with coordinates scaled to original size.
        """
        processed_detections = []

        orig_w, orig_h = original_image_size
        det_w, det_h = self.detection_image_size

        # Calculate scaling factors
        scale_x = orig_w / det_w
        scale_y = orig_h / det_h

        for detection in detections:
            processed_det = detection.copy()

            # Scale bounding box coordinates
            if 'bbox' in detection:
                bbox = detection['bbox']
                processed_det['bbox'] = [
                    bbox[0] * scale_x,  # x_min
                    bbox[1] * scale_y,  # y_min
                    bbox[2] * scale_x,  # x_max
                    bbox[3] * scale_y   # y_max
                ]

            processed_detections.append(processed_det)

        return processed_detections

    def denormalize_recognition_image(
        self,
        tensor: torch.Tensor
    ) -> np.ndarray:
        """Denormalize recognition image tensor back to displayable format.

        Args:
            tensor: Normalized image tensor.

        Returns:
            Denormalized image in BGR format.
        """
        # Convert tensor to numpy
        if tensor.dim() == 4:
            # Batch dimension present, take first image
            tensor = tensor[0]

        # Move to CPU and convert to numpy
        image = tensor.cpu().numpy()

        # Transpose from CHW to HWC
        image = np.transpose(image, (1, 2, 0))

        # Denormalize
        image = denormalize_image(
            image,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

        return image

    def create_detection_mosaic(
        self,
        images: List[np.ndarray],
        labels: Optional[List[List[Dict]]] = None
    ) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """Create mosaic augmentation for detection training.

        Args:
            images: List of 4 images for mosaic.
            labels: Optional list of labels for each image.

        Returns:
            Tuple of (mosaic_image, mosaic_labels).
        """
        if len(images) != 4:
            raise ValueError("Mosaic requires exactly 4 images")

        # Resize all images to half the target size
        half_size = (
            self.detection_image_size[0] // 2,
            self.detection_image_size[1] // 2
        )

        resized_images = []
        for img in images:
            resized = resize_image(img, half_size, maintain_aspect_ratio=True)
            resized_images.append(resized)

        # Create mosaic
        top_row = np.hstack([resized_images[0], resized_images[1]])
        bottom_row = np.hstack([resized_images[2], resized_images[3]])
        mosaic = np.vstack([top_row, bottom_row])

        # Adjust labels if provided
        mosaic_labels = None
        if labels is not None:
            mosaic_labels = []

            # Offsets for each quadrant
            offsets = [
                (0, 0),                                    # Top-left
                (half_size[0], 0),                        # Top-right
                (0, half_size[1]),                        # Bottom-left
                (half_size[0], half_size[1])              # Bottom-right
            ]

            for i, (img_labels, (offset_x, offset_y)) in enumerate(zip(labels, offsets)):
                for label in img_labels:
                    adjusted_label = label.copy()

                    # Adjust bounding box coordinates
                    if 'bbox' in label:
                        bbox = label['bbox']
                        adjusted_label['bbox'] = [
                            bbox[0] + offset_x,
                            bbox[1] + offset_y,
                            bbox[2] + offset_x,
                            bbox[3] + offset_y
                        ]

                    mosaic_labels.append(adjusted_label)

        return mosaic, mosaic_labels