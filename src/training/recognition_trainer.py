"""Simplified training pipeline for recognition model."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from tqdm import tqdm

from ..utils.logging import LoggerMixin
from ..data.dataset import MonkeyDataset
from ..recognition.model import create_recognition_model
from ..recognition.database import FeatureDatabase


class RecognitionTrainer(LoggerMixin):
    """Simplified trainer for monkey recognition model."""

    def __init__(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        num_classes: int = 10,
        backbone: str = 'resnet50',
        embedding_size: int = 512,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = 'auto',
        experiment_tracker: Optional[Any] = None
    ):
        """Initialize recognition trainer.

        Args:
            train_dir: Training data directory.
            val_dir: Validation data directory (optional).
            num_classes: Number of monkey classes.
            backbone: Model backbone architecture.
            embedding_size: Feature embedding size.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            device: Device to use for training.
            experiment_tracker: Experiment tracker instance.
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.num_classes = num_classes
        self.backbone = backbone
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.experiment_tracker = experiment_tracker

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.logger.info(f"Using device: {self.device}")

        # Training state
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.training_history = []
        self.best_accuracy = 0.0

    def _create_data_loaders(self):
        """Create training and validation data loaders."""
        # Create training dataset
        train_dataset = MonkeyDataset(self.train_dir, training=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        # Create validation dataset if directory exists
        val_loader = None
        if self.val_dir and os.path.exists(self.val_dir):
            val_dataset = MonkeyDataset(self.val_dir, training=False)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )

        return train_loader, val_loader

    def _create_model(self):
        """Create recognition model."""
        self.model = create_recognition_model(
            num_classes=self.num_classes,
            backbone=self.backbone,
            embedding_size=self.embedding_size,
            margin_loss='arcface',
            pretrained=True
        )
        self.model.to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create loss criterion
        self.criterion = nn.CrossEntropyLoss()

        self.logger.info(f"Created model with {self.num_classes} classes")

    def _train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def _validate_epoch(self, val_loader):
        """Validate for one epoch."""
        if val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(
        self,
        epochs: int = 100,
        save_dir: str = "models",
        model_name: str = "recognition_model"
    ) -> str:
        """Train the recognition model.

        Args:
            epochs: Number of training epochs.
            save_dir: Directory to save the model.
            model_name: Name of the model file.

        Returns:
            Path to the saved model.
        """
        self.logger.info(f"Starting training for {epochs} epochs")

        # Create data loaders
        train_loader, val_loader = self._create_data_loaders()

        # Create model
        self._create_model()

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Training loop
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc = self._validate_epoch(val_loader)

            # Log metrics
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }

            self.training_history.append(metrics)

            # Log to experiment tracker
            if self.experiment_tracker:
                self.experiment_tracker.log_metrics(metrics, epoch=epoch)

            # Print progress
            if val_loader:
                self.logger.info(
                    f'Epoch {epoch}/{epochs}: '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
                )
            else:
                self.logger.info(
                    f'Epoch {epoch}/{epochs}: '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%'
                )

            # Save best model
            current_acc = val_acc if val_loader else train_acc
            if current_acc > self.best_accuracy:
                self.best_accuracy = current_acc
                model_path = os.path.join(save_dir, f"{model_name}_best.pt")

                # Save model state
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': current_acc,
                    'num_classes': self.num_classes,
                    'config': {
                        'backbone': self.backbone,
                        'embedding_size': self.embedding_size,
                        'num_classes': self.num_classes
                    }
                }, model_path)

                self.logger.info(f"Saved best model: {model_path} (accuracy: {current_acc:.2f}%)")

                # Save checkpoint to experiment tracker
                if self.experiment_tracker:
                    self.experiment_tracker.save_checkpoint(
                        self.model, self.optimizer, epoch, metrics, is_best=True
                    )

        # Create feature database
        self._create_feature_database(train_loader, save_dir)

        self.logger.info("Training completed!")
        return model_path

    def _create_feature_database(self, train_loader, save_dir):
        """Create feature database from training data."""
        self.logger.info("Creating feature database...")

        database = FeatureDatabase()
        self.model.eval()

        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Extracting features"):
                images = images.to(self.device)

                # Extract features
                features = self.model.extract_features(images)
                features_np = features.cpu().numpy()

                # Add to database
                for i, (feature, label) in enumerate(zip(features_np, labels)):
                    monkey_id = f"monkey_{label.item():03d}"
                    database.add_feature(monkey_id, feature)

        # Save database
        db_path = os.path.join(save_dir, "../databases", "features.pkl")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        database.save(db_path)

        self.logger.info(f"Feature database saved: {db_path}")

    def get_final_metrics(self) -> Dict[str, float]:
        """Get final training metrics."""
        if not self.training_history:
            return {}

        final_epoch = self.training_history[-1]
        return {
            'final_train_accuracy': final_epoch.get('train_accuracy', 0.0),
            'final_val_accuracy': final_epoch.get('val_accuracy', 0.0),
            'best_accuracy': self.best_accuracy,
            'total_epochs': len(self.training_history)
        }