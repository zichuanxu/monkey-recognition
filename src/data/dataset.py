"""Dataset classes for loading monkey images."""

import os
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Callable, Dict
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from ..utils.image_utils import load_image
from ..utils.logging import LoggerMixin


class MonkeyDataset(Dataset, LoggerMixin):
    """Dataset for loading monkey images organized by monkey ID."""

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        training: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        """Initialize monkey dataset.

        Args:
            data_dir: Directory containing monkey subdirectories.
            transform: Optional transform to apply to images.
            training: Whether this is for training (affects augmentation).
            image_size: Target image size for resizing.
            extensions: Valid image file extensions.
        """
        self.data_dir = data_dir
        self.training = training
        self.image_size = image_size
        self.extensions = extensions

        # Load data
        self.image_paths, self.labels, self.class_names = self._load_data()
        self.num_classes = len(self.class_names)

        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transform(training)
        else:
            self.transform = transform

        self.logger.info(f"Loaded {len(self.image_paths)} images from {self.num_classes} monkey classes")

    def _load_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Load image paths and labels from directory structure.

        Returns:
            Tuple of (image_paths, labels, class_names).
        """
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")

        image_paths = []
        labels = []
        class_names = []

        # Get all subdirectories (monkey IDs)
        for item in sorted(os.listdir(self.data_dir)):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                class_names.append(item)

        if not class_names:
            raise ValueError(f"No subdirectories found in {self.data_dir}")

        # Load images from each class directory
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(self.data_dir, class_name)

            for filename in os.listdir(class_dir):
                if any(filename.lower().endswith(ext) for ext in self.extensions):
                    image_path = os.path.join(class_dir, filename)
                    image_paths.append(image_path)
                    labels.append(class_idx)

        if not image_paths:
            raise ValueError(f"No images found in {self.data_dir}")

        return image_paths, labels, class_names

    def _get_default_transform(self, training: bool = True) -> transforms.Compose:
        """Get default image transforms.

        Returns:
            Torchvision transform pipeline.
        """
        if training:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index.

        Args:
            idx: Item index.

        Returns:
            Tuple of (image_tensor, label).
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = load_image(image_path)
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            # Return a black image as fallback
            image = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)

        # Convert BGR to RGB and to PIL Image
        image = image[:, :, ::-1]  # BGR to RGB
        image = Image.fromarray(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self) -> List[str]:
        """Get list of class names (monkey IDs)."""
        return self.class_names.copy()