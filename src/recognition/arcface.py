"""ArcFace loss implementation for face recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

from ..utils.logging import LoggerMixin


class ArcFace(nn.Module, LoggerMixin):
    """ArcFace loss implementation.

    Reference: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    https://arxiv.org/abs/1801.07698
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        """Initialize ArcFace layer.

        Args:
            in_features: Input feature dimension.
            out_features: Number of classes (monkeys).
            scale: Scale factor for the cosine similarity.
            margin: Angular margin penalty.
            easy_margin: Whether to use easy margin.
        """
        super(ArcFace, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Initialize weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cosine and sine of margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.logger.info(
            f"ArcFace initialized: in_features={in_features}, out_features={out_features}, "
            f"scale={scale}, margin={margin}"
        )

    def forward(
        self,
        input: torch.Tensor,
        label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of ArcFace layer.

        Args:
            input: Input features of shape (batch_size, in_features).
            label: Ground truth labels of shape (batch_size,). Required for training.

        Returns:
            Output logits of shape (batch_size, out_features).
        """
        # Normalize input features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if label is None:
            # Inference mode - return scaled cosine similarity
            return self.scale * cosine

        # Training mode - apply angular margin
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding for labels
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply margin only to the correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

    def extract_features(self, input: torch.Tensor) -> torch.Tensor:
        """Extract normalized features without classification.

        Args:
            input: Input features of shape (batch_size, in_features).

        Returns:
            Normalized features of shape (batch_size, in_features).
        """
        return F.normalize(input)


class AdaCos(nn.Module, LoggerMixin):
    """AdaCos loss implementation as an alternative to ArcFace.

    Reference: AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations
    https://arxiv.org/abs/1905.00292
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        margin: float = 0.5,
        theta_zero: float = math.pi / 4
    ):
        """Initialize AdaCos layer.

        Args:
            in_features: Input feature dimension.
            out_features: Number of classes.
            margin: Fixed margin.
            theta_zero: Initial theta value.
        """
        super(AdaCos, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.theta_zero = theta_zero

        # Initialize weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Adaptive scale parameter
        self.scale = math.sqrt(2) * math.log(out_features - 1)

        self.logger.info(
            f"AdaCos initialized: in_features={in_features}, out_features={out_features}, "
            f"margin={margin}, initial_scale={self.scale:.4f}"
        )

    def forward(
        self,
        input: torch.Tensor,
        label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of AdaCos layer.

        Args:
            input: Input features.
            label: Ground truth labels.

        Returns:
            Output logits.
        """
        # Normalize input features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if label is None:
            return self.scale * cosine

        # Calculate theta
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        # Create one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply margin to positive samples
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.scale * cosine), torch.zeros_like(cosine))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            self.scale = torch.log(B_avg) / torch.cos(torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med))

        output = self.scale * cosine
        return output


class CosFace(nn.Module, LoggerMixin):
    """CosFace loss implementation.

    Reference: CosFace: Large Margin Cosine Loss for Deep Face Recognition
    https://arxiv.org/abs/1801.09414
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 30.0,
        margin: float = 0.4
    ):
        """Initialize CosFace layer.

        Args:
            in_features: Input feature dimension.
            out_features: Number of classes.
            scale: Scale factor.
            margin: Cosine margin.
        """
        super(CosFace, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin

        # Initialize weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.logger.info(
            f"CosFace initialized: in_features={in_features}, out_features={out_features}, "
            f"scale={scale}, margin={margin}"
        )

    def forward(
        self,
        input: torch.Tensor,
        label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of CosFace layer.

        Args:
            input: Input features.
            label: Ground truth labels.

        Returns:
            Output logits.
        """
        # Normalize input features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if label is None:
            return self.scale * cosine

        # Create one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply margin to positive samples
        output = cosine - one_hot * self.margin
        output *= self.scale

        return output


class SphereFace(nn.Module, LoggerMixin):
    """SphereFace loss implementation.

    Reference: SphereFace: Deep Hypersphere Embedding for Face Recognition
    https://arxiv.org/abs/1704.08063
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        margin: int = 4
    ):
        """Initialize SphereFace layer.

        Args:
            in_features: Input feature dimension.
            out_features: Number of classes.
            margin: Angular margin (integer).
        """
        super(SphereFace, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin

        # Initialize weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute coefficients for multiple angle formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

        self.logger.info(
            f"SphereFace initialized: in_features={in_features}, out_features={out_features}, "
            f"margin={margin}"
        )

    def forward(
        self,
        input: torch.Tensor,
        label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of SphereFace layer.

        Args:
            input: Input features.
            label: Ground truth labels.

        Returns:
            Output logits.
        """
        # Normalize weights
        w_norm = F.normalize(self.weight, dim=1)

        # Calculate cosine similarity
        cosine = F.linear(input, w_norm)

        if label is None:
            return cosine

        # Calculate phi (multiple angle)
        cosine_m = self.mlambda[self.margin](cosine)

        # Create one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply angular margin
        output = one_hot * cosine_m + (1.0 - one_hot) * cosine

        return output


def create_margin_loss(
    loss_type: str,
    in_features: int,
    out_features: int,
    **kwargs
) -> nn.Module:
    """Create margin-based loss function.

    Args:
        loss_type: Type of loss ('arcface', 'cosface', 'adacos', 'sphereface').
        in_features: Input feature dimension.
        out_features: Number of classes.
        **kwargs: Additional arguments for specific loss functions.

    Returns:
        Margin loss module.
    """
    loss_type = loss_type.lower()

    if loss_type == 'arcface':
        return ArcFace(
            in_features=in_features,
            out_features=out_features,
            scale=kwargs.get('scale', 30.0),
            margin=kwargs.get('margin', 0.5),
            easy_margin=kwargs.get('easy_margin', False)
        )
    elif loss_type == 'cosface':
        return CosFace(
            in_features=in_features,
            out_features=out_features,
            scale=kwargs.get('scale', 30.0),
            margin=kwargs.get('margin', 0.4)
        )
    elif loss_type == 'adacos':
        return AdaCos(
            in_features=in_features,
            out_features=out_features,
            margin=kwargs.get('margin', 0.5),
            theta_zero=kwargs.get('theta_zero', math.pi / 4)
        )
    elif loss_type == 'sphereface':
        return SphereFace(
            in_features=in_features,
            out_features=out_features,
            margin=kwargs.get('margin', 4)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance.

    Reference: Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class.
            gamma: Focusing parameter.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of Focal Loss.

        Args:
            input: Predicted logits.
            target: Ground truth labels.

        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CenterLoss(nn.Module):
    """Center Loss implementation for intra-class compactness.

    Reference: A Discriminative Feature Learning Approach for Deep Face Recognition
    https://ydwen.github.io/papers/WenECCV16.pdf
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        use_gpu: bool = True
    ):
        """Initialize Center Loss.

        Args:
            num_classes: Number of classes.
            feat_dim: Feature dimension.
            use_gpu: Whether to use GPU.
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of Center Loss.

        Args:
            x: Input features.
            labels: Ground truth labels.

        Returns:
            Center loss value.
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss