"""Annotation generation for YOLO detection training."""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import yaml
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

from ..utils.image_utils import load_image, get_image_files, get_image_info
from ..utils.file_utils import ensure_dir, save_yaml
from ..utils.data_structures import BoundingBox
from ..utils.logging import LoggerMixin


class YOLOAnnotationGenerator(LoggerMixin):
    """Generate YOLO format annotations from monkey face dataset."""

    def __init__(self, face_margin: float = 0.1, min_face_size: int = 32):
        """Initialize annotation generator.

        Args:
            face_margin: Margin around detected/assumed face area (as fraction of image size).
            min_face_size: Minimum face size in pixels.
        """
        self.face_margin = face_margin
        self.min_face_size = min_face_size
        self.class_names = ['monkey_face']  # Single class for face detection

    def generate_annotations_from_directory(
        self,
        data_dir: str,
        output_dir: str,
        val_split: float = 0.2,
        assume_full_face: bool = True,
        face_detection_method: str = 'full_image'
    ) -> Dict[str, str]:
        """Generate YOLO annotations from monkey dataset directory.

        Args:
            data_dir: Directory containing monkey subdirectories.
            output_dir: Output directory for YOLO dataset.
            val_split: Validation split ratio.
            assume_full_face: Whether to assume entire image contains a face.
            face_detection_method: Method for face detection ('full_image', 'center_crop', 'opencv_face').

        Returns:
            Dictionary with paths to generated dataset components.
        """
        self.logger.info(f"Generating YOLO annotations from {data_dir}")

        # Create output directory structure
        dataset_paths = self._create_yolo_structure(output_dir)

        # Collect all image paths and their monkey IDs
        all_images = []
        for monkey_id in os.listdir(data_dir):
            monkey_dir = os.path.join(data_dir, monkey_id)
            if not os.path.isdir(monkey_dir):
                continue

            images = get_image_files(monkey_dir)
            for img_path in images:
                all_images.append((img_path, monkey_id))

        if len(all_images) == 0:
            raise ValueError(f"No images found in {data_dir}")

        self.logger.info(f"Found {len(all_images)} images")

        # Split into train and validation
        train_images, val_images = train_test_split(
            all_images, test_size=val_split, random_state=42, stratify=[x[1] for x in all_images]
        )

        # Process training images
        self.logger.info(f"Processing {len(train_images)} training images")
        self._process_image_set(
            train_images,
            dataset_paths['train_images'],
            dataset_paths['train_labels'],
            face_detection_method,
            assume_full_face
        )

        # Process validation images
        self.logger.info(f"Processing {len(val_images)} validation images")
        self._process_image_set(
            val_images,
            dataset_paths['val_images'],
            dataset_paths['val_labels'],
            face_detection_method,
            assume_full_face
        )

        # Create dataset configuration file
        self._create_dataset_config(dataset_paths['config'], dataset_paths)

        self.logger.info(f"YOLO dataset generated successfully in {output_dir}")
        return dataset_paths

    def _create_yolo_structure(self, output_dir: str) -> Dict[str, str]:
        """Create YOLO dataset directory structure.

        Args:
            output_dir: Output directory path.

        Returns:
            Dictionary with paths to dataset components.
        """
        paths = {
            'root': output_dir,
            'train_images': os.path.join(output_dir, 'images', 'train'),
            'val_images': os.path.join(output_dir, 'images', 'val'),
            'train_labels': os.path.join(output_dir, 'labels', 'train'),
            'val_labels': os.path.join(output_dir, 'labels', 'val'),
            'config': os.path.join(output_dir, 'dataset.yaml')
        }

        # Create directories
        for path in paths.values():
            if path.endswith('.yaml'):
                ensure_dir(os.path.dirname(path))
            else:
                ensure_dir(path)

        return paths

    def _process_image_set(
        self,
        images: List[Tuple[str, str]],
        images_dir: str,
        labels_dir: str,
        face_detection_method: str,
        assume_full_face: bool
    ) -> None:
        """Process a set of images and generate annotations.

        Args:
            images: List of (image_path, monkey_id) tuples.
            images_dir: Directory to copy images to.
            labels_dir: Directory to save labels to.
            face_detection_method: Face detection method.
            assume_full_face: Whether to assume full face in image.
        """
        for i, (img_path, monkey_id) in enumerate(images):
            try:
                # Load image
                image = load_image(img_path)
                if image is None:
                    self.logger.warning(f"Failed to load image: {img_path}")
                    continue

                # Generate bounding box
                bbox = self._generate_face_bbox(
                    image, face_detection_method, assume_full_face
                )

                if bbox is None:
                    self.logger.warning(f"No face detected in: {img_path}")
                    continue

                # Copy image to dataset
                img_filename = f"{monkey_id}_{i:06d}{Path(img_path).suffix}"
                img_dst_path = os.path.join(images_dir, img_filename)
                shutil.copy2(img_path, img_dst_path)

                # Create label file
                label_filename = f"{monkey_id}_{i:06d}.txt"
                label_path = os.path.join(labels_dir, label_filename)
                self._save_yolo_annotation(label_path, bbox, image.shape[:2])

            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")
                continue

    def _generate_face_bbox(
        self,
        image: np.ndarray,
        method: str,
        assume_full_face: bool
    ) -> Optional[BoundingBox]:
        """Generate face bounding box using specified method.

        Args:
            image: Input image.
            method: Detection method.
            assume_full_face: Whether to assume full face.

        Returns:
            BoundingBox or None if no face detected.
        """
        h, w = image.shape[:2]

        if method == 'full_image' or assume_full_face:
            # Assume entire image contains a face with some margin
            margin_x = int(w * self.face_margin)
            margin_y = int(h * self.face_margin)

            x_min = max(0, margin_x)
            y_min = max(0, margin_y)
            x_max = min(w, w - margin_x)
            y_max = min(h, h - margin_y)

            return BoundingBox(x_min, y_min, x_max, y_max, 1.0)

        elif method == 'center_crop':
            # Use center portion of image as face
            crop_size = min(w, h) * (1 - 2 * self.face_margin)
            crop_size = max(crop_size, self.min_face_size)

            center_x, center_y = w // 2, h // 2
            half_size = int(crop_size // 2)

            x_min = max(0, center_x - half_size)
            y_min = max(0, center_y - half_size)
            x_max = min(w, center_x + half_size)
            y_max = min(h, center_y + half_size)

            return BoundingBox(x_min, y_min, x_max, y_max, 1.0)

        elif method == 'opencv_face':
            # Use OpenCV face detection as fallback
            return self._detect_face_opencv(image)

        else:
            raise ValueError(f"Unknown face detection method: {method}")

    def _detect_face_opencv(self, image: np.ndarray) -> Optional[BoundingBox]:
        """Detect face using OpenCV Haar cascades.

        Args:
            image: Input image.

        Returns:
            BoundingBox or None if no face detected.
        """
        try:
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size)
            )

            if len(faces) > 0:
                # Take the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face

                return BoundingBox(x, y, x + w, y + h, 1.0)

            return None

        except Exception as e:
            self.logger.error(f"OpenCV face detection failed: {e}")
            return None

    def _save_yolo_annotation(
        self,
        label_path: str,
        bbox: BoundingBox,
        image_shape: Tuple[int, int]
    ) -> None:
        """Save bounding box in YOLO format.

        Args:
            label_path: Path to save label file.
            bbox: Bounding box to save.
            image_shape: Image shape (height, width).
        """
        h, w = image_shape

        # Convert to YOLO format (normalized center coordinates)
        x_center_norm, y_center_norm, width_norm, height_norm = bbox.to_yolo_format(w, h)

        # YOLO format: class_id x_center y_center width height
        yolo_line = f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\\n"

        with open(label_path, 'w') as f:
            f.write(yolo_line)

    def _create_dataset_config(self, config_path: str, dataset_paths: Dict[str, str]) -> None:
        """Create YOLO dataset configuration file.

        Args:
            config_path: Path to save config file.
            dataset_paths: Dictionary with dataset paths.
        """
        config = {
            'path': dataset_paths['root'],
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,  # Number of classes
            'names': self.class_names
        }

        save_yaml(config, config_path)
        self.logger.info(f"Dataset config saved to {config_path}")

    def validate_annotations(self, dataset_dir: str) -> Dict[str, any]:
        """Validate generated YOLO annotations.

        Args:
            dataset_dir: Path to YOLO dataset directory.

        Returns:
            Validation results dictionary.
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Check dataset structure
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        for dir_name in required_dirs:
            dir_path = os.path.join(dataset_dir, dir_name)
            if not os.path.exists(dir_path):
                results['valid'] = False
                results['errors'].append(f"Missing directory: {dir_name}")

        # Check config file
        config_path = os.path.join(dataset_dir, 'dataset.yaml')
        if not os.path.exists(config_path):
            results['valid'] = False
            results['errors'].append("Missing dataset.yaml config file")

        if not results['valid']:
            return results

        # Validate train and val sets
        for split in ['train', 'val']:
            split_results = self._validate_split(dataset_dir, split)
            results['statistics'][split] = split_results

            if not split_results['valid']:
                results['valid'] = False
                results['errors'].extend([
                    f"{split}: {error}" for error in split_results['errors']
                ])
                results['warnings'].extend([
                    f"{split}: {warning}" for warning in split_results['warnings']
                ])

        return results

    def _validate_split(self, dataset_dir: str, split: str) -> Dict[str, any]:
        """Validate a specific dataset split.

        Args:
            dataset_dir: Dataset directory path.
            split: Split name ('train' or 'val').

        Returns:
            Validation results for this split.
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'image_count': 0,
            'label_count': 0,
            'matched_pairs': 0
        }

        images_dir = os.path.join(dataset_dir, 'images', split)
        labels_dir = os.path.join(dataset_dir, 'labels', split)

        # Get image and label files
        image_files = get_image_files(images_dir)
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

        results['image_count'] = len(image_files)
        results['label_count'] = len(label_files)

        # Check for matching pairs
        image_stems = {Path(f).stem for f in image_files}
        label_stems = {Path(f).stem for f in label_files}

        matched_stems = image_stems & label_stems
        results['matched_pairs'] = len(matched_stems)

        # Check for mismatches
        images_without_labels = image_stems - label_stems
        labels_without_images = label_stems - image_stems

        if images_without_labels:
            results['warnings'].append(
                f"{len(images_without_labels)} images without labels"
            )

        if labels_without_images:
            results['warnings'].append(
                f"{len(labels_without_images)} labels without images"
            )

        # Validate label format
        invalid_labels = []
        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            if not self._validate_label_file(label_path):
                invalid_labels.append(label_file)

        if invalid_labels:
            results['errors'].append(f"Invalid label files: {invalid_labels}")
            results['valid'] = False

        return results

    def _validate_label_file(self, label_path: str) -> bool:
        """Validate YOLO label file format.

        Args:
            label_path: Path to label file.

        Returns:
            True if valid, False otherwise.
        """
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    return False

                # Check class ID
                class_id = int(parts[0])
                if class_id != 0:  # We only have one class
                    return False

                # Check coordinates (should be normalized)
                for coord in parts[1:]:
                    coord_val = float(coord)
                    if coord_val < 0 or coord_val > 1:
                        return False

            return True

        except Exception:
            return False


def create_detection_dataset(
    data_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    face_margin: float = 0.1,
    face_detection_method: str = 'full_image'
) -> Dict[str, str]:
    """Create YOLO detection dataset from monkey images.

    Args:
        data_dir: Source data directory with monkey subdirectories.
        output_dir: Output directory for YOLO dataset.
        val_split: Validation split ratio.
        face_margin: Margin around face area.
        face_detection_method: Method for face detection.

    Returns:
        Dictionary with dataset paths.
    """
    generator = YOLOAnnotationGenerator(face_margin=face_margin)

    return generator.generate_annotations_from_directory(
        data_dir=data_dir,
        output_dir=output_dir,
        val_split=val_split,
        assume_full_face=True,
        face_detection_method=face_detection_method
    )