"""Recognition model architecture with various backbones."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from typing import Optional, Dict, Any, Tuple

from .arcface import ArcFace, create_margin_loss
from ..utils.logging import LoggerMixin


class MonkeyRecognitionModel(nn.Module, LoggerMixin):
    """Monkey face recognition model with configurable backbone and margin loss."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet50',
        embedding_size: int = 512,
        margin_loss: str = 'arcface',
        margin_loss_params: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        dropout: float = 0.0,
        bn_freeze: bool = False
    ):
        """Initialize recognition model.

        Args:
            num_classes: Number of monkey classes.
            backbone: Backbone architecture name.
            embedding_size: Size of feature embeddings.
            margin_loss: Type of margin loss ('arcface', 'cosface', etc.).
            margin_loss_params: Parameters for margin loss.
            pretrained: Whether to use pretrained backbone.
            dropout: Dropout rate.
            bn_freeze: Whether to freeze batch normalization layers.
        """
        super(MonkeyRecognitionModel, self).__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone
        self.embedding_size = embedding_size
        self.margin_loss_type = margin_loss
        self.dropout_rate = dropout

        # Initialize backbone
        self.backbone = self._create_backbone(backbone, pretrained)
        backbone_features = self._get_backbone_features(backbone)

        # Feature projection layer
        self.neck = nn.Sequential(
            nn.Linear(backbone_features, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

        # Dropout layer
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Margin loss layer
        margin_params = margin_loss_params or {}
        self.margin_loss = create_margin_loss(
            margin_loss,
            embedding_size,
            num_classes,
            **margin_params
        )

        # Freeze batch normalization if requested
        if bn_freeze:
            self._freeze_bn()

        self.logger.info(
            f"MonkeyRecognitionModel initialized: backbone={backbone}, "
            f"embedding_size={embedding_size}, num_classes={num_classes}, "
            f"margin_loss={margin_loss}"
        )

    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Create backbone network.

        Args:
            backbone: Backbone architecture name.
            pretrained: Whether to use pretrained weights.

        Returns:
            Backbone network.
        """
        backbone = backbone.lower()

        if backbone.startswith('resnet'):
            if backbone == 'resnet18':
                model = models.resnet18(pretrained=pretrained)
            elif backbone == 'resnet34':
                model = models.resnet34(pretrained=pretrained)
            elif backbone == 'resnet50':
                model = models.resnet50(pretrained=pretrained)
            elif backbone == 'resnet101':
                model = models.resnet101(pretrained=pretrained)
            elif backbone == 'resnet152':
                model = models.resnet152(pretrained=pretrained)
            else:
                raise ValueError(f"Unknown ResNet variant: {backbone}")

            # Remove final classification layer
            return nn.Sequential(*list(model.children())[:-1])

        elif backbone.startswith('efficientnet'):
            try:
                model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
                return model
            except Exception as e:
                raise ValueError(f"Failed to create EfficientNet {backbone}: {e}")

        elif backbone.startswith('mobilenet'):
            if backbone == 'mobilenet_v2':
                model = models.mobilenet_v2(pretrained=pretrained)
                return model.features
            elif backbone == 'mobilenet_v3_small':
                model = models.mobilenet_v3_small(pretrained=pretrained)
                return model.features
            elif backbone == 'mobilenet_v3_large':
                model = models.mobilenet_v3_large(pretrained=pretrained)
                return model.features
            else:
                raise ValueError(f"Unknown MobileNet variant: {backbone}")

        elif backbone.startswith('vit'):
            try:
                model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
                return model
            except Exception as e:
                raise ValueError(f"Failed to create Vision Transformer {backbone}: {e}")

        elif backbone.startswith('swin'):
            try:
                model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
                return model
            except Exception as e:
                raise ValueError(f"Failed to create Swin Transformer {backbone}: {e}")

        else:
            # Try to create with timm
            try:
                model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
                return model
            except Exception as e:
                raise ValueError(f"Unknown backbone architecture: {backbone}. Error: {e}")

    def _get_backbone_features(self, backbone: str) -> int:
        """Get number of features from backbone.

        Args:
            backbone: Backbone architecture name.

        Returns:
            Number of output features.
        """
        backbone = backbone.lower()

        # Common feature dimensions
        feature_dims = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048,
            'efficientnet_b0': 1280,
            'efficientnet_b1': 1280,
            'efficientnet_b2': 1408,
            'efficientnet_b3': 1536,
            'efficientnet_b4': 1792,
            'efficientnet_b5': 2048,
            'efficientnet_b6': 2304,
            'efficientnet_b7': 2560,
            'mobilenet_v2': 1280,
            'mobilenet_v3_small': 576,
            'mobilenet_v3_large': 960,
            'vit_base_patch16_224': 768,
            'vit_large_patch16_224': 1024,
            'swin_base_patch4_window7_224': 1024,
            'swin_large_patch4_window7_224': 1536,
        }

        if backbone in feature_dims:
            return feature_dims[backbone]

        # Try to infer from model
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                features = self.backbone(dummy_input)
                if len(features.shape) == 4:  # Conv features
                    return features.shape[1]
                elif len(features.shape) == 2:  # Linear features
                    return features.shape[1]
                else:
                    raise ValueError(f"Unexpected feature shape: {features.shape}")
        except Exception as e:
            raise ValueError(f"Could not determine feature dimension for {backbone}: {e}")

    def _freeze_bn(self) -> None:
        """Freeze batch normalization layers."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input images of shape (batch_size, 3, height, width).
            labels: Ground truth labels for training.

        Returns:
            Output logits or features depending on mode.
        """
        # Extract backbone features
        features = self.backbone(x)

        # Global average pooling if needed
        if len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)

        # Project to embedding space
        embeddings = self.neck(features)

        # Apply dropout
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)

        # Apply margin loss
        output = self.margin_loss(embeddings, labels)

        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized feature embeddings.

        Args:
            x: Input images.

        Returns:
            Normalized feature embeddings.
        """
        # Extract backbone features
        features = self.backbone(x)

        # Global average pooling if needed
        if len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)

        # Project to embedding space
        embeddings = self.neck(features)

        # Normalize features
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        return normalized_embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model information.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'backbone': self.backbone_name,
            'embedding_size': self.embedding_size,
            'num_classes': self.num_classes,
            'margin_loss': self.margin_loss_type,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.logger.info("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.logger.info("Backbone unfrozen")

    def freeze_layers(self, num_layers: int) -> None:
        """Freeze first N layers of the backbone.

        Args:
            num_layers: Number of layers to freeze.
        """
        if self.backbone_name.startswith('resnet'):
            layers = list(self.backbone.children())
            for i, layer in enumerate(layers[:num_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
            self.logger.info(f"Frozen first {num_layers} layers of ResNet")
        else:
            self.logger.warning(f"Layer freezing not implemented for {self.backbone_name}")


class EnsembleModel(nn.Module, LoggerMixin):
    """Ensemble of multiple recognition models."""

    def __init__(
        self,
        models: list,
        weights: Optional[list] = None,
        voting_method: str = 'average'
    ):
        """Initialize ensemble model.

        Args:
            models: List of recognition models.
            weights: Optional weights for each model.
            voting_method: Voting method ('average', 'weighted', 'max').
        """
        super(EnsembleModel, self).__init__()

        self.models = nn.ModuleList(models)
        self.voting_method = voting_method

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            self.weights = weights

        self.logger.info(f"Ensemble model initialized with {len(models)} models")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble.

        Args:
            x: Input images.

        Returns:
            Ensemble predictions.
        """
        outputs = []

        for model in self.models:
            with torch.no_grad():
                output = model(x)
                outputs.append(output)

        # Combine outputs
        if self.voting_method == 'average':
            ensemble_output = torch.stack(outputs).mean(dim=0)
        elif self.voting_method == 'weighted':
            weighted_outputs = [w * out for w, out in zip(self.weights, outputs)]
            ensemble_output = torch.stack(weighted_outputs).sum(dim=0)
        elif self.voting_method == 'max':
            ensemble_output = torch.stack(outputs).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")

        return ensemble_output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract ensemble features.

        Args:
            x: Input images.

        Returns:
            Ensemble features.
        """
        features = []

        for model in self.models:
            with torch.no_grad():
                feat = model.extract_features(x)
                features.append(feat)

        # Average features
        ensemble_features = torch.stack(features).mean(dim=0)
        return F.normalize(ensemble_features, p=2, dim=1)


def create_recognition_model(
    num_classes: int,
    backbone: str = 'resnet50',
    embedding_size: int = 512,
    margin_loss: str = 'arcface',
    pretrained: bool = True,
    **kwargs
) -> MonkeyRecognitionModel:
    """Create recognition model with specified configuration.

    Args:
        num_classes: Number of monkey classes.
        backbone: Backbone architecture.
        embedding_size: Embedding dimension.
        margin_loss: Type of margin loss.
        pretrained: Whether to use pretrained backbone.
        **kwargs: Additional arguments.

    Returns:
        Recognition model instance.
    """
    return MonkeyRecognitionModel(
        num_classes=num_classes,
        backbone=backbone,
        embedding_size=embedding_size,
        margin_loss=margin_loss,
        pretrained=pretrained,
        **kwargs
    )