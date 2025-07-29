"""Training pipeline for YOLOv8 detection model."""

import os
import yaml
import torch
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO
from pathlib import Path
import shutil
from datetime import datetime
import json

from ..utils.config import Config, DetectionConfig
from ..utils.logging import LoggerMixin
from ..utils.file_utils import ensure_dir, save_json
from ..utils.data_structures import TrainingMetrics, ModelInfo
from ..data.annotation_generator import create_detection_dataset
from ..detection.detector import MonkeyFaceDetector
from ..detection.utils import DetectionUtils


class YOLOTrainer(LoggerMixin):
    """Trainer for YOLOv8 monkey face detection model."""

    def __init__(
        self,
        config: Config,
        output_dir: str = "outputs/detection_training"
    ):
        """Initialize YOLO trainer.

Args:
            config: Training configuration.
            output_dir: Output directory for training artifacts.
        """
        self.config = config
        self.detection_config = config.detection
        self.output_dir = output_dir

        # Create output directories
        ensure_dir(self.output_dir)
        self.models_dir = os.path.join(self.output_dir, "models")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.plots_dir = os.path.join(self.output_dir, "plots")

        for dir_path in [self.models_dir, self.logs_dir, self.plots_dir]:
            ensure_dir(dir_path)

        # Training state
        self.model = None
        self.training_results = None
        self.best_model_path = None

        self.logger.info(f"YOLOTrainer initialized with output dir: {self.output_dir}")

    def prepare_dataset(
        self,
        data_dir: str,
        force_regenerate: bool = False
    ) -> str:
        """Prepare YOLO dataset from monkey images.

        Args:
            data_dir: Directory containing monkey subdirectories.
            force_regenerate: Whether to regenerate dataset if it exists.

        Returns:
            Path to generated dataset directory.
        """
        dataset_dir = os.path.join(self.output_dir, "dataset")

        # Check if dataset already exists
        if os.path.exists(dataset_dir) and not force_regenerate:
            dataset_config_path = os.path.join(dataset_dir, "dataset.yaml")
            if os.path.exists(dataset_config_path):
                self.logger.info(f"Using existing dataset: {dataset_dir}")
                return dataset_dir

        self.logger.info(f"Preparing YOLO dataset from {data_dir}")

        # Generate YOLO dataset
        dataset_paths = create_detection_dataset(
            data_dir=data_dir,
            output_dir=dataset_dir,
            val_split=0.2,
            face_margin=0.1,
            face_detection_method='full_image'
        )

        self.logger.info(f"Dataset prepared successfully: {dataset_dir}")
        return dataset_dir

    def train(
        self,
        dataset_path: str,
        resume: bool = False,
        pretrained_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train YOLOv8 detection model.

        Args:
            dataset_path: Path to YOLO dataset directory.
            resume: Whether to resume training from checkpoint.
            pretrained_model: Path to pretrained model. If None, uses default.

        Returns:
            Training results dictionary.
        """
        self.logger.info("Starting YOLOv8 training")

        # Validate dataset
        dataset_config_path = os.path.join(dataset_path, "dataset.yaml")
        if not os.path.exists(dataset_config_path):
            raise ValueError(f"Dataset config not found: {dataset_config_path}")

        # Initialize model
        model_name = pretrained_model or self.detection_config.model_name
        self.logger.info(f"Loading model: {model_name}")

        try:
            self.model = YOLO(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise

        # Set up training parameters
        train_params = self._get_training_params(dataset_config_path)

        # Start training
        try:
            self.logger.info("Starting training with parameters:")
            for key, value in train_params.items():
                self.logger.info(f"  {key}: {value}")

            results = self.model.train(**train_params)

            # Store training results
            self.training_results = results
            self.best_model_path = os.path.join(
                train_params['project'],
                train_params['name'],
                'weights',
                'best.pt'
            )

            # Save training summary
            self._save_training_summary(results, train_params)

            self.logger.info(f"Training completed. Best model: {self.best_model_path}")

            return {
                'success': True,
                'best_model_path': self.best_model_path,
                'results': results,
                'training_params': train_params
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _get_training_params(self, dataset_config_path: str) -> Dict[str, Any]:
        """Get training parameters for YOLO.

        Args:
            dataset_config_path: Path to dataset configuration.

        Returns:
            Training parameters dictionary.
        """
        training_config = self.detection_config.training

        # Create unique experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"monkey_detection_{timestamp}"

        params = {
            'data': dataset_config_path,
            'epochs': training_config.epochs,
            'batch': training_config.batch_size,
            'imgsz': self.detection_config.image_size,
            'lr0': training_config.learning_rate,
            'patience': training_config.patience,
            'save_period': training_config.save_period,
            'project': self.models_dir,
            'name': experiment_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'seed': self.config.system.seed,
            'deterministic': True,
            'single_cls': True,  # Single class (monkey face)
            'rect': False,  # Rectangular training
            'cos_lr': True,  # Cosine learning rate scheduler
            'close_mosaic': 10,  # Close mosaic augmentation in last N epochs
            'resume': False,  # Will be set if resuming
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,  # Dataset fraction to use
            'profile': False,  # Profile ONNX and TensorRT speeds
            'freeze': None,  # Freeze layers: backbone=10, first3=0:3
            'multi_scale': False,  # Multi-scale training
            'overlap_mask': True,  # Overlap masks
            'mask_ratio': 4,  # Mask downsample ratio
            'dropout': 0.0,  # Use dropout regularization
            'val': True,  # Validate/test during training
        }

        # Add device configuration
        if self.config.system.device != 'auto':
            params['device'] = self.config.system.device

        return params

    def _save_training_summary(
        self,
        results,
        train_params: Dict[str, Any]
    ) -> None:
        """Save training summary and metrics.

        Args:
            results: Training results from YOLO.
            train_params: Training parameters used.
        """
        summary = {
            'training_completed': datetime.now().isoformat(),
            'training_params': train_params,
            'config': {
                'detection': {
                    'model_name': self.detection_config.model_name,
                    'image_size': self.detection_config.image_size,
                    'confidence_threshold': self.detection_config.confidence_threshold,
                    'iou_threshold': self.detection_config.iou_threshold
                },
                'system': {
                    'device': self.config.system.device,
                    'seed': self.config.system.seed
                }
            }
        }

        # Add results if available
        if hasattr(results, 'results_dict'):
            summary['final_metrics'] = results.results_dict

        # Save summary
        summary_path = os.path.join(self.logs_dir, 'training_summary.json')
        save_json(summary, summary_path)

        self.logger.info(f"Training summary saved: {summary_path}")

    def evaluate(
        self,
        model_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Evaluate trained detection model.

        Args:
            model_path: Path to model to evaluate. If None, uses best trained model.
            test_data_path: Path to test dataset. If None, uses validation set.
            save_results: Whether to save evaluation results.

        Returns:
            Evaluation results dictionary.
        """
        # Determine model path
        if model_path is None:
            if self.best_model_path is None:
                raise ValueError("No trained model available. Train first or provide model_path.")
            model_path = self.best_model_path

        if not os.path.exists(model_path):
            raise ValueError(f"Model not found: {model_path}")

        self.logger.info(f"Evaluating model: {model_path}")

        try:
            # Load model
            model = YOLO(model_path)

            # Run validation
            if test_data_path:
                # Use custom test data
                results = model.val(data=test_data_path, split='test')
            else:
                # Use validation split from training
                results = model.val()

            # Extract metrics
            metrics = self._extract_evaluation_metrics(results)

            if save_results:
                self._save_evaluation_results(metrics, model_path)

            self.logger.info("Evaluation completed successfully")
            self.logger.info(f"mAP@0.5: {metrics.get('mAP_50', 0.0):.4f}")
            self.logger.info(f"mAP@0.5:0.95: {metrics.get('mAP_50_95', 0.0):.4f}")

            return {
                'success': True,
                'metrics': metrics,
                'model_path': model_path
            }

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_evaluation_metrics(self, results) -> Dict[str, float]:
        """Extract evaluation metrics from YOLO results.

        Args:
            results: YOLO validation results.

        Returns:
            Metrics dictionary.
        """
        metrics = {}

        try:
            # Extract metrics from results
            if hasattr(results, 'results_dict'):
                results_dict = results.results_dict

                # Map YOLO metric names to our format
                metric_mapping = {
                    'metrics/mAP50(B)': 'mAP_50',
                    'metrics/mAP50-95(B)': 'mAP_50_95',
                    'metrics/precision(B)': 'precision',
                    'metrics/recall(B)': 'recall',
                    'val/box_loss': 'box_loss',
                    'val/cls_loss': 'cls_loss',
                    'val/dfl_loss': 'dfl_loss'
                }

                for yolo_key, our_key in metric_mapping.items():
                    if yolo_key in results_dict:
                        metrics[our_key] = float(results_dict[yolo_key])

            # Add additional computed metrics
            if 'precision' in metrics and 'recall' in metrics:
                p, r = metrics['precision'], metrics['recall']
                if p + r > 0:
                    metrics['f1_score'] = 2 * p * r / (p + r)
                else:
                    metrics['f1_score'] = 0.0

        except Exception as e:
            self.logger.warning(f"Failed to extract some metrics: {e}")

        return metrics

    def _save_evaluation_results(
        self,
        metrics: Dict[str, float],
        model_path: str
    ) -> None:
        """Save evaluation results.

        Args:
            metrics: Evaluation metrics.
            model_path: Path to evaluated model.
        """
        results = {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': model_path,
            'metrics': metrics,
            'config': {
                'confidence_threshold': self.detection_config.confidence_threshold,
                'iou_threshold': self.detection_config.iou_threshold,
                'image_size': self.detection_config.image_size
            }
        }

        # Save results
        results_path = os.path.join(self.logs_dir, 'evaluation_results.json')
        save_json(results, results_path)

        self.logger.info(f"Evaluation results saved: {results_path}")

    def export_model(
        self,
        model_path: Optional[str] = None,
        export_format: str = 'onnx',
        output_path: Optional[str] = None
    ) -> str:
        """Export trained model to different format.

        Args:
            model_path: Path to model to export. If None, uses best trained model.
            export_format: Export format ('onnx', 'torchscript', 'tflite', etc.).
            output_path: Output path for exported model.

        Returns:
            Path to exported model.
        """
        # Determine model path
        if model_path is None:
            if self.best_model_path is None:
                raise ValueError("No trained model available. Train first or provide model_path.")
            model_path = self.best_model_path

        if not os.path.exists(model_path):
            raise ValueError(f"Model not found: {model_path}")

        self.logger.info(f"Exporting model to {export_format} format")

        try:
            # Load model
            model = YOLO(model_path)

            # Export model
            export_path = model.export(
                format=export_format,
                imgsz=self.detection_config.image_size
            )

            # Move to desired location if specified
            if output_path:
                ensure_dir(os.path.dirname(output_path))
                shutil.move(export_path, output_path)
                export_path = output_path

            self.logger.info(f"Model exported successfully: {export_path}")
            return export_path

        except Exception as e:
            self.logger.error(f"Model export failed: {e}")
            raise

    def create_detector(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None
    ) -> MonkeyFaceDetector:
        """Create detector instance from trained model.

        Args:
            model_path: Path to trained model. If None, uses best trained model.
            confidence_threshold: Detection confidence threshold.

        Returns:
            MonkeyFaceDetector instance.
        """
        # Determine model path
        if model_path is None:
            if self.best_model_path is None:
                raise ValueError("No trained model available. Train first or provide model_path.")
            model_path = self.best_model_path

        # Use config threshold if not specified
        if confidence_threshold is None:
            confidence_threshold = self.detection_config.confidence_threshold

        detector = MonkeyFaceDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=self.detection_config.iou_threshold,
            max_detections=self.detection_config.max_detections,
            device=self.config.system.device
        )

        self.logger.info(f"Detector created with model: {model_path}")
        return detector

    def get_training_info(self) -> Dict[str, Any]:
        """Get information about training session.

        Returns:
            Training information dictionary.
        """
        info = {
            'output_dir': self.output_dir,
            'models_dir': self.models_dir,
            'logs_dir': self.logs_dir,
            'best_model_path': self.best_model_path,
            'training_completed': self.training_results is not None
        }

        # Add config info
        info['config'] = {
            'model_name': self.detection_config.model_name,
            'image_size': self.detection_config.image_size,
            'epochs': self.detection_config.training.epochs,
            'batch_size': self.detection_config.training.batch_size,
            'learning_rate': self.detection_config.training.learning_rate
        }

        return info

    def cleanup_training_artifacts(self, keep_best_model: bool = True) -> None:
        """Clean up training artifacts to save space.

        Args:
            keep_best_model: Whether to keep the best model file.
        """
        self.logger.info("Cleaning up training artifacts")

        # Find training run directories
        if os.path.exists(self.models_dir):
            for run_dir in os.listdir(self.models_dir):
                run_path = os.path.join(self.models_dir, run_dir)
                if not os.path.isdir(run_path):
                    continue

                weights_dir = os.path.join(run_path, 'weights')
                if os.path.exists(weights_dir):
                    # Remove all weights except best.pt if keeping best model
                    for weight_file in os.listdir(weights_dir):
                        if keep_best_model and weight_file == 'best.pt':
                            continue

                        weight_path = os.path.join(weights_dir, weight_file)
                        if os.path.isfile(weight_path):
                            os.remove(weight_path)
                            self.logger.debug(f"Removed: {weight_path}")

                # Remove other large directories
                for subdir in ['runs', 'wandb']:
                    subdir_path = os.path.join(run_path, subdir)
                    if os.path.exists(subdir_path):
                        shutil.rmtree(subdir_path)
                        self.logger.debug(f"Removed directory: {subdir_path}")

        self.logger.info("Training artifacts cleanup completed")