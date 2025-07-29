"""Post-processing utilities for detection results."""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import cv2

from ..utils.data_structures import BoundingBox
from ..utils.logging import LoggerMixin


class DetectionPostProcessor(LoggerMixin):
    """Post-processor for detection results with NMS and filtering."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        min_box_size: int = 32,
        max_box_size: Optional[int] = None,
        aspect_ratio_range: Tuple[float, float] = (.0)
    ):
        """Initialize detection post-processor.

        Args:
            confidence_threshold: Minimum confidence for detections.
            iou_threshold: IoU threshold for NMS.
            min_box_size: Minimum bounding box size (width or height).
            max_box_size: Maximum bounding box size (width or height).
            aspect_ratio_range: Valid aspect ratio range (min, max).
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size
        self.aspect_ratio_range = aspect_ratio_range

    def apply_nms(
        self,
        detections: List[BoundingBox],
        iou_threshold: Optional[float] = None
    ) -> List[BoundingBox]:
        """Apply Non-Maximum Suppression to detections.

        Args:
            detections: List of bounding box detections.
            iou_threshold: IoU threshold for NMS. If None, uses instance threshold.

        Returns:
            Filtered list of detections after NMS.
        """
        if len(detections) == 0:
            return []

        iou_thresh = iou_threshold or self.iou_threshold

        # Convert to format expected by OpenCV NMS
        boxes = []
        scores = []

        for detection in detections:
            boxes.append([
                detection.x_min,
                detection.y_min,
                detection.width,
                detection.height
            ])
            scores.append(detection.confidence)

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            iou_thresh
        )

        # Return filtered detections
        if len(indices) == 0:
            return []

        filtered_detections = []
        for i in indices.flatten():
            filtered_detections.append(detections[i])

        self.logger.debug(f"NMS: {len(detections)} -> {len(filtered_detections)} detections")
        return filtered_detections

    def filter_by_confidence(
        self,
        detections: List[BoundingBox],
        threshold: Optional[float] = None
    ) -> List[BoundingBox]:
        """Filter detections by confidence threshold.

        Args:
            detections: List of detections.
            threshold: Confidence threshold. If None, uses instance threshold.

        Returns:
            Filtered detections.
        """
        conf_thresh = threshold or self.confidence_threshold

        filtered = [d for d in detections if d.confidence >= conf_thresh]

        self.logger.debug(
            f"Confidence filter: {len(detections)} -> {len(filtered)} detections "
            f"(threshold: {conf_thresh})"
        )

        return filtered

    def filter_by_size(
        self,
        detections: List[BoundingBox],
        min_size: Optional[int] = None,
        max_size: Optional[int] = None
    ) -> List[BoundingBox]:
        """Filter detections by bounding box size.

        Args:
            detections: List of detections.
            min_size: Minimum size (width or height).
            max_size: Maximum size (width or height).

        Returns:
            Filtered detections.
        """
        min_sz = min_size or self.min_box_size
        max_sz = max_size or self.max_box_size

        filtered = []
        for detection in detections:
            width = detection.width
            height = detection.height

            # Check minimum size
            if min(width, height) < min_sz:
                continue

            # Check maximum size
            if max_sz is not None and max(width, height) > max_sz:
                continue

            filtered.append(detection)

        self.logger.debug(
            f"Size filter: {len(detections)} -> {len(filtered)} detections "
            f"(min: {min_sz}, max: {max_sz})"
        )

        return filtered

    def filter_by_aspect_ratio(
        self,
        detections: List[BoundingBox],
        aspect_ratio_range: Optional[Tuple[float, float]] = None
    ) -> List[BoundingBox]:
        """Filter detections by aspect ratio.

        Args:
            detections: List of detections.
            aspect_ratio_range: Valid aspect ratio range (min, max).

        Returns:
            Filtered detections.
        """
        ar_range = aspect_ratio_range or self.aspect_ratio_range
        min_ar, max_ar = ar_range

        filtered = []
        for detection in detections:
            aspect_ratio = detection.width / detection.height

            if min_ar <= aspect_ratio <= max_ar:
                filtered.append(detection)

        self.logger.debug(
            f"Aspect ratio filter: {len(detections)} -> {len(filtered)} detections "
            f"(range: {ar_range})"
        )

        return filtered

    def filter_by_image_bounds(
        self,
        detections: List[BoundingBox],
        image_shape: Tuple[int, int],
        margin: int = 0
    ) -> List[BoundingBox]:
        """Filter detections that are within image bounds.

        Args:
            detections: List of detections.
            image_shape: Image shape (height, width).
            margin: Margin from image edges.

        Returns:
            Filtered detections.
        """
        h, w = image_shape

        filtered = []
        for detection in detections:
            # Check if detection is within bounds
            if (detection.x_min >= margin and
                detection.y_min >= margin and
                detection.x_max <= w - margin and
                detection.y_max <= h - margin):
                filtered.append(detection)

        self.logger.debug(
            f"Bounds filter: {len(detections)} -> {len(filtered)} detections"
        )

        return filtered

    def remove_overlapping_detections(
        self,
        detections: List[BoundingBox],
        overlap_threshold: float = 0.8
    ) -> List[BoundingBox]:
        """Remove highly overlapping detections, keeping the one with highest confidence.

        Args:
            detections: List of detections.
            overlap_threshold: IoU threshold for considering detections as overlapping.

        Returns:
            Filtered detections with overlaps removed.
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence (descending)
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        filtered = []
        for detection in sorted_detections:
            # Check if this detection overlaps significantly with any already selected
            is_overlapping = False

            for selected in filtered:
                iou = detection.iou(selected)
                if iou > overlap_threshold:
                    is_overlapping = True
                    break

            if not is_overlapping:
                filtered.append(detection)

        self.logger.debug(
            f"Overlap removal: {len(detections)} -> {len(filtered)} detections "
            f"(threshold: {overlap_threshold})"
        )

        return filtered

    def process_detections(
        self,
        detections: List[BoundingBox],
        image_shape: Optional[Tuple[int, int]] = None,
        apply_nms: bool = True,
        apply_size_filter: bool = True,
        apply_aspect_ratio_filter: bool = True,
        apply_bounds_filter: bool = True,
        remove_overlaps: bool = True
    ) -> List[BoundingBox]:
        """Apply complete post-processing pipeline to detections.

        Args:
            detections: Raw detections from model.
            image_shape: Image shape for bounds filtering.
            apply_nms: Whether to apply NMS.
            apply_size_filter: Whether to apply size filtering.
            apply_aspect_ratio_filter: Whether to apply aspect ratio filtering.
            apply_bounds_filter: Whether to apply bounds filtering.
            remove_overlaps: Whether to remove overlapping detections.

        Returns:
            Post-processed detections.
        """
        if len(detections) == 0:
            return []

        processed = detections.copy()

        # Apply confidence filtering first
        processed = self.filter_by_confidence(processed)

        # Apply size filtering
        if apply_size_filter:
            processed = self.filter_by_size(processed)

        # Apply aspect ratio filtering
        if apply_aspect_ratio_filter:
            processed = self.filter_by_aspect_ratio(processed)

        # Apply bounds filtering
        if apply_bounds_filter and image_shape is not None:
            processed = self.filter_by_image_bounds(processed, image_shape)

        # Apply NMS
        if apply_nms:
            processed = self.apply_nms(processed)

        # Remove overlapping detections
        if remove_overlaps:
            processed = self.remove_overlapping_detections(processed)

        self.logger.info(
            f"Post-processing: {len(detections)} -> {len(processed)} detections"
        )

        return processed

    def calculate_detection_statistics(
        self,
        detections: List[BoundingBox]
    ) -> Dict[str, Any]:
        """Calculate statistics for a set of detections.

        Args:
            detections: List of detections.

        Returns:
            Statistics dictionary.
        """
        if len(detections) == 0:
            return {
                'count': 0,
                'avg_confidence': 0.0,
                'confidence_std': 0.0,
                'avg_area': 0.0,
                'avg_aspect_ratio': 0.0
            }

        confidences = [d.confidence for d in detections]
        areas = [d.area for d in detections]
        aspect_ratios = [d.width / d.height for d in detections]

        stats = {
            'count': len(detections),
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'avg_area': np.mean(areas),
            'area_std': np.std(areas),
            'min_area': np.min(areas),
            'max_area': np.max(areas),
            'avg_aspect_ratio': np.mean(aspect_ratios),
            'aspect_ratio_std': np.std(aspect_ratios),
            'min_aspect_ratio': np.min(aspect_ratios),
            'max_aspect_ratio': np.max(aspect_ratios)
        }

        return stats

    def merge_close_detections(
        self,
        detections: List[BoundingBox],
        distance_threshold: float = 50.0,
        confidence_strategy: str = 'max'
    ) -> List[BoundingBox]:
        """Merge detections that are close to each other.

        Args:
            detections: List of detections.
            distance_threshold: Maximum distance between centers to merge.
            confidence_strategy: How to handle confidence ('max', 'avg', 'weighted_avg').

        Returns:
            List of merged detections.
        """
        if len(detections) <= 1:
            return detections

        merged = []
        used = set()

        for i, detection in enumerate(detections):
            if i in used:
                continue

            # Find nearby detections
            group = [detection]
            group_indices = [i]

            for j, other in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue

                # Calculate distance between centers
                center1 = detection.center
                center2 = other.center
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

                if distance <= distance_threshold:
                    group.append(other)
                    group_indices.append(j)

            # Mark as used
            used.update(group_indices)

            # Merge group into single detection
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged_detection = self._merge_detection_group(group, confidence_strategy)
                merged.append(merged_detection)

        self.logger.debug(
            f"Merge close: {len(detections)} -> {len(merged)} detections "
            f"(threshold: {distance_threshold})"
        )

        return merged

    def _merge_detection_group(
        self,
        group: List[BoundingBox],
        confidence_strategy: str
    ) -> BoundingBox:
        """Merge a group of detections into a single detection.

        Args:
            group: List of detections to merge.
            confidence_strategy: Confidence merging strategy.

        Returns:
            Merged detection.
        """
        # Calculate merged bounding box
        x_mins = [d.x_min for d in group]
        y_mins = [d.y_min for d in group]
        x_maxs = [d.x_max for d in group]
        y_maxs = [d.y_max for d in group]

        merged_x_min = min(x_mins)
        merged_y_min = min(y_mins)
        merged_x_max = max(x_maxs)
        merged_y_max = max(y_maxs)

        # Calculate merged confidence
        confidences = [d.confidence for d in group]

        if confidence_strategy == 'max':
            merged_confidence = max(confidences)
        elif confidence_strategy == 'avg':
            merged_confidence = np.mean(confidences)
        elif confidence_strategy == 'weighted_avg':
            # Weight by area
            areas = [d.area for d in group]
            total_area = sum(areas)
            if total_area > 0:
                merged_confidence = sum(c * a for c, a in zip(confidences, areas)) / total_area
            else:
                merged_confidence = np.mean(confidences)
        else:
            merged_confidence = max(confidences)

        return BoundingBox(
            merged_x_min,
            merged_y_min,
            merged_x_max,
            merged_y_max,
            merged_confidence
        )