"""Configuration management utilities."""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""
    train_dir: str = "data/train_Magface"
    test_dir: str = "data/test_image"
    output_dir: str = "outputs"
    models_dir: str = "models"


@dataclass
class DetectionTrainingConfig:
    """Detection training configuration."""
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    patience: int = 10
    save_period: int = 10


@dataclass
class DetectionConfig:
    """Detection model configuration."""
    model_name: str = "yolov8m.pt"
    image_size: int = 640
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    training: DetectionTrainingConfig = field(default_factory=DetectionTrainingConfig)


@dataclass
class RecognitionTrainingConfig:
    """Recognition training configuration."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    patience: int = 15


@dataclass
class RecognitionDatabaseConfig:
    """Recognition database configuration."""
    similarity_threshold: float = 0.6
    max_features_per_monkey: int = 10


@dataclass
class RecognitionConfig:
    """Recognition model configuration."""
    backbone: str = "resnet50"
    embedding_size: int = 512
    margin: float = 0.5
    scale: float = 30.0
    training: RecognitionTrainingConfig = field(default_factory=RecognitionTrainingConfig)
    database: RecognitionDatabaseConfig = field(default_factory=RecognitionDatabaseConfig)


@dataclass
class SystemConfig:
    """System configuration."""
    device: str = "auto"
    num_workers: int = 4
    seed: int = 42
    log_level: str = "INFO"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: list = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])
    save_predictions: bool = True
    save_visualizations: bool = True


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


class ConfigManager:
    """Configuration manager for loading and saving configurations."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path or "config/default_config.yaml"
        self.config = self.load_config()

    def load_config(self) -> Config:
        """Load configuration from YAML file.

        Returns:
            Config object with loaded settings.
        """
        if not os.path.exists(self.config_path):
            print(f"Config file {self.config_path} not found. Using default configuration.")
            return Config()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)

            return self._dict_to_config(config_dict)

        except Exception as e:
            print(f"Error loading config file {self.config_path}: {e}")
            print("Using default configuration.")
            return Config()

    def save_config(self, config: Config, save_path: Optional[str] = None) -> None:
        """Save configuration to YAML file.

        Args:
            config: Configuration object to save.
            save_path: Path to save configuration. If None, uses current config_path.
        """
        save_path = save_path or self.config_path

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        config_dict = self._config_to_dict(config)

        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object."""
        config = Config()

        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])

        if 'detection' in config_dict:
            detection_dict = config_dict['detection'].copy()
            if 'training' in detection_dict:
                training_config = DetectionTrainingConfig(**detection_dict.pop('training'))
                detection_dict['training'] = training_config
            config.detection = DetectionConfig(**detection_dict)

        if 'recognition' in config_dict:
            recognition_dict = config_dict['recognition'].copy()
            if 'training' in recognition_dict:
                training_config = RecognitionTrainingConfig(**recognition_dict.pop('training'))
                recognition_dict['training'] = training_config
            if 'database' in recognition_dict:
                database_config = RecognitionDatabaseConfig(**recognition_dict.pop('database'))
                recognition_dict['database'] = database_config
            config.recognition = RecognitionConfig(**recognition_dict)

        if 'system' in config_dict:
            config.system = SystemConfig(**config_dict['system'])

        if 'evaluation' in config_dict:
            config.evaluation = EvaluationConfig(**config_dict['evaluation'])

        return config

    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary."""
        return {
            'data': {
                'train_dir': config.data.train_dir,
                'test_dir': config.data.test_dir,
                'output_dir': config.data.output_dir,
                'models_dir': config.data.models_dir,
            },
            'detection': {
                'model_name': config.detection.model_name,
                'image_size': config.detection.image_size,
                'confidence_threshold': config.detection.confidence_threshold,
                'iou_threshold': config.detection.iou_threshold,
                'max_detections': config.detection.max_detections,
                'training': {
                    'epochs': config.detection.training.epochs,
                    'batch_size': config.detection.training.batch_size,
                    'learning_rate': config.detection.training.learning_rate,
                    'patience': config.detection.training.patience,
                    'save_period': config.detection.training.save_period,
                }
            },
            'recognition': {
                'backbone': config.recognition.backbone,
                'embedding_size': config.recognition.embedding_size,
                'margin': config.recognition.margin,
                'scale': config.recognition.scale,
                'training': {
                    'epochs': config.recognition.training.epochs,
                    'batch_size': config.recognition.training.batch_size,
                    'learning_rate': config.recognition.training.learning_rate,
                    'weight_decay': config.recognition.training.weight_decay,
                    'patience': config.recognition.training.patience,
                },
                'database': {
                    'similarity_threshold': config.recognition.database.similarity_threshold,
                    'max_features_per_monkey': config.recognition.database.max_features_per_monkey,
                }
            },
            'system': {
                'device': config.system.device,
                'num_workers': config.system.num_workers,
                'seed': config.system.seed,
                'log_level': config.system.log_level,
            },
            'evaluation': {
                'metrics': config.evaluation.metrics,
                'save_predictions': config.evaluation.save_predictions,
                'save_visualizations': config.evaluation.save_visualizations,
            }
        }


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration object.

    Args:
        config_path: Path to configuration file.

    Returns:
        Config object.
    """
    manager = ConfigManager(config_path)
    return manager.config