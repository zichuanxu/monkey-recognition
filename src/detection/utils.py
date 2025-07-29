"""Utility functions for detection module."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import os
from pathlib import Path

from ..utils.data_structures import BoundingBox
from ..utils.image_utils import load_image, save_image
from ..utils.logging import LoggerMixin


class DetectionUtils(LoggerMixin):
    """Utility functions for detection operations."""

    @staticmethod
    def calculate_iou_matrix(
        detections1: List[BoundingBox],
        detections2: List[BoundingBox]
    ) -> np.ndarray:
        """Calculate IoU matrix between two sets of detections.

        Args:
            detections1: First set of detections.
            detections2: Second set of detections.

        Returns:
         IoU matrix of shape (len(detections1), len(detections2)).
        """
        if len(detections1) == 0 or len(detections2) == 0:
            return np.zeros((len(detections1), len(detections2)))

        iou_matrix = np.zeros((len(detections1), len(detections2)))

        for i, det1 in enumerate(detections1):
            for j, det2 in enumerate(detections2):
                iou_matrix[i, j] = det1.iou(det2)

        return iou_matrix

    @staticmethod
    def match_detections(
        pred_detections: List[BoundingBox],
        gt_detections: List[BoundingBox],
        iou_threshold: float = 0.5
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match predicted detections with ground truth detections.

        Args:
            pred_detections: Predicted detections.
            gt_detections: Ground truth detections.
            iou_threshold: IoU threshold for matching.

        Returns:
            Tuple of (matches, unmatched_preds, unmatched_gts).
            matches: List of (pred_idx, gt_idx) pairs.
            unmatched_preds: List of unmatched prediction indices.
            unmatched_gts: List of unmatched ground truth indices.
        """
        if len(pred_detections) == 0:
            return [], [], list(range(len(gt_detections)))

        if len(gt_detections) == 0:
            return [], list(range(len(pred_detections))), []

        # Calculate IoU matrix
        iou_matrix = DetectionUtils.calculate_iou_matrix(pred_detections, gt_detections)

        # Find matches using greedy assignment
        matches = []
        used_preds = set()
        used_gts = set()

        # Sort by IoU (descending)
        candidates = []
        for i in range(len(pred_detections)):
            for j in range(len(gt_detections)):
                if iou_matrix[i, j] >= iou_threshold:
                    candidates.append((iou_matrix[i, j], i, j))

        candidates.sort(reverse=True)

        # Assign matches
        for iou, pred_idx, gt_idx in candidates:
            if pred_idx not in used_preds and gt_idx not in used_gts:
                matches.append((pred_idx, gt_idx))
                used_preds.add(pred_idx)
                used_gts.add(gt_idx)

        # Find unmatched detections
        unmatched_preds = [i for i in range(len(pred_detections)) if i not in used_preds]
        unmatched_gts = [i for i in range(len(gt_detections)) if i not in used_gts]

        return matches, unmatched_preds, unmatched_gts

    @staticmethod
    def calculate_detection_metrics(
        pred_detections: List[BoundingBox],
        gt_detections: List[BoundingBox],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate detection metrics (precision, recall, F1).

        Args:
            pred_detections: Predicted detections.
            gt_detections: Ground truth detections.
            iou_threshold: IoU threshold for matching.

        Returns:
            Dictionary with metrics.
        """
        matches, unmatched_preds, unmatched_gts = DetectionUtils.match_detections(
            pred_detections, gt_detections, iou_threshold
        )

        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }

    @staticmethod
    def calculate_map(
        all_pred_detections: List[List[BoundingBox]],
        all_gt_detections: List[List[BoundingBox]],
        iou_thresholds: List[float] = None
    ) -> Dict[str, float]:
        """Calculate mean Average Precision (mAP) across multiple images.

        Args:
            all_pred_detections: List of prediction lists for each image.
            all_gt_detections: List of ground truth lists for each image.
            iou_thresholds: IoU thresholds to evaluate. Default: [0.5:0.95:0.05].

        Returns:
            Dictionary with mAP metrics.
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95

        if len(all_pred_detections) != len(all_gt_detections):
            raise ValueError("Number of prediction and ground truth lists must match")

        # Collect all detections with image indices
        all_preds = []
        all_gts = []

        for img_idx, (preds, gts) in enumerate(zip(all_pred_detections, all_gt_detections)):
            for pred in preds:
                all_preds.append((pred, img_idx))
            for gt in gts:
                all_gts.append((gt, img_idx))

        # Sort predictions by confidence (descending)
        all_preds.sort(key=lambda x: x[0].confidence, reverse=True)

        # Calculate AP for each IoU threshold
        aps = []

        for iou_thresh in iou_thresholds:
            tp = np.zeros(len(all_preds))
            fp = np.zeros(len(all_preds))

            # Track which ground truths have been matched
            gt_matched = set()

            for pred_idx, (pred_det, pred_img_idx) in enumerate(all_preds):
                # Find ground truths in the same image
                img_gts = [(gt_det, gt_idx) for gt_idx, (gt_det, gt_img_idx) in enumerate(all_gts)
                          if gt_img_idx == pred_img_idx]

                best_iou = 0.0
                best_gt_idx = -1

                for gt_det, gt_idx in img_gts:
                    if gt_idx in gt_matched:
                        continue

                    iou = pred_det.iou(gt_det)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_thresh:
                    tp[pred_idx] = 1
                    gt_matched.add(best_gt_idx)
                else:
                    fp[pred_idx] = 1

            # Calculate precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / len(all_gts) if len(all_gts) > 0 else np.zeros_like(tp_cumsum)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            precisions = np.nan_to_num(precisions)

            # Calculate AP using 11-point interpolation
            ap = DetectionUtils._calculate_ap_11_point(precisions, recalls)
            aps.append(ap)

        # Calculate mAP metrics
        map_50 = aps[0] if len(aps) > 0 else 0.0  # mAP@0.5
        map_50_95 = np.mean(aps) if len(aps) > 0 else 0.0  # mAP@0.5:0.95

        return {
            'mAP@0.5': map_50,
            'mAP@0.5:0.95': map_50_95,
            'APs': aps,
            'IoU_thresholds': iou_thresholds
        }

    @staticmethod
    def _calculate_ap_11_point(precisions: np.ndarray, recalls: np.ndarray) -> float:
        """Calculate Average Precision using 11-point interpolation.

        Args:
            precisions: Precision values.
            recalls: Recall values.

        Returns:
            Average Precision value.
        """
        # 11-point interpolation
        recall_thresholds = np.linspace(0, 1, 11)
        interpolated_precisions = []

        for recall_thresh in recall_thresholds:
            # Find precisions at recalls >= threshold
            valid_precisions = precisions[recalls >= recall_thresh]
            if len(valid_precisions) > 0:
                interpolated_precisions.append(np.max(valid_precisions))
            else:
                interpolated_precisions.append(0.0)

        return np.mean(interpolated_precisions)

    @staticmethod
    def save_detection_results(
        image_paths: List[str],
        detections_list: List[List[BoundingBox]],
        output_dir: str,
        save_crops: bool = False,
        save_visualizations: bool = True,
        crop_padding: int = 10
    ) -> None:
        """Save detection results to files.

        Args:
            image_paths: List of image paths.
            detections_list: List of detection lists for each image.
            output_dir: Output directory.
            save_crops: Whether to save cropped faces.
            save_visualizations: Whether to save visualization images.
            crop_padding: Padding for face crops.
        """
        logger = DetectionUtils().logger

        os.makedirs(output_dir, exist_ok=True)

        if save_crops:
            crops_dir = os.path.join(output_dir, 'crops')
            os.makedirs(crops_dir, exist_ok=True)

        if save_visualizations:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

        results = []

        for img_path, detections in zip(image_paths, detections_list):
            img_name = Path(img_path).stem
            image = load_image(img_path)

            if image is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            # Save detection info
            img_results = {
                'image_path': img_path,
                'detections': []
            }

            for i, detection in enumerate(detections):
                det_info = {
                    'bbox': [detection.x_min, detection.y_min, detection.x_max, detection.y_max],
                    'confidence': detection.confidence,
                    'area': detection.area
                }
                img_results['detections'].append(det_info)

                # Save crop
                if save_crops:
                    h, w = image.shape[:2]
                    x1 = max(0, detection.x_min - crop_padding)
                    y1 = max(0, detection.y_min - crop_padding)
                    x2 = min(w, detection.x_max + crop_padding)
                    y2 = min(h, detection.y_max + crop_padding)

                    crop = image[y1:y2, x1:x2]
                    crop_path = os.path.join(crops_dir, f"{img_name}_face_{i:03d}.jpg")
                    save_image(crop, crop_path)

            results.append(img_results)

            # Save visualization
            if save_visualizations:
                vis_image = DetectionUtils.draw_detections(image, detections)
                vis_path = os.path.join(vis_dir, f"{img_name}_detections.jpg")
                save_image(vis_image, vis_path)

        # Save results JSON
        import json
        results_path = os.path.join(output_dir, 'detection_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Detection results saved to {output_dir}")

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[BoundingBox],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = True
    ) -> np.ndarray:
        """Draw detections on image.

        Args:
            image: Input image.
            detections: List of detections.
            color: Bounding box color.
            thickness: Line thickness.
            show_confidence: Whether to show confidence scores.

        Returns:
            Image with drawn detections.
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

            # Draw confidence
            if show_confidence:
                label = f"{detection.confidence:.2f}"

                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness
                )

                # Draw background
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

    @staticmethod
    def create_detection_grid(
        images: List[np.ndarray],
        detections_list: List[List[BoundingBox]],
        grid_size: Tuple[int, int] = (2, 2),
        image_size: Tuple[int, int] = (300, 300)
    ) -> np.ndarray:
        """Create a grid of images with detections.

        Args:
            images: List of images.
            detections_list: List of detection lists.
            grid_size: Grid size (rows, cols).
            image_size: Size for each image in grid.

        Returns:
            Grid image.
        """
        rows, cols = grid_size
        img_h, img_w = image_size

        # Create grid canvas
        grid_image = np.zeros((rows * img_h, cols * img_w, 3), dtype=np.uint8)

        for i in range(min(len(images), rows * cols)):
            row = i // cols
            col = i % cols

            # Resize image
            resized_img = cv2.resize(images[i], (img_w, img_h))

            # Draw detections (scaled)
            if i < len(detections_list):
                orig_h, orig_w = images[i].shape[:2]
                scale_x = img_w / orig_w
                scale_y = img_h / orig_h

                for detection in detections_list[i]:
                    # Scale coordinates
                    x1 = int(detection.x_min * scale_x)
                    y1 = int(detection.y_min * scale_y)
                    x2 = int(detection.x_max * scale_x)
                    y2 = int(detection.y_max * scale_y)

                    # Draw bounding box
                    cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw confidence
                    label = f"{detection.confidence:.2f}"
                    cv2.putText(
                        resized_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )

            # Place in grid
            y_start = row * img_h
            y_end = (row + 1) * img_h
            x_start = col * img_w
            x_end = (col + 1) * img_w

            grid_image[y_start:y_end, x_start:x_end] = resized_img

        return grid_image