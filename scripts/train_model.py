#!/usr/bin/env python3
"""Simple training script for monkey face recognition system."""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.recognition_trainer import RecognitionTrainer
from src.training.experiment_tracker import ExperimentTracker
from src.utils.logging import setup_logging


def count_classes(data_dir):
    """Count number of monkey classes in training data."""
    if not os.path.exists(data_dir):
        raise ValueError(f"Training directory not found: {data_dir}")

    classes = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]
    return len(classes), classes


def main():
    parser = argparse.ArgumentParser(description="Train monkey face recognition model")

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/training_config.yaml',
        help='Path to training configuration file'
    )

    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='monkey_recognition',
        help='Experiment name'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level='INFO')

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Please create a configuration file or use the default template.")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Count classes in training data
    train_dir = config['data']['train_dir']
    try:
        num_classes, class_names = count_classes(train_dir)
        print(f"Found {num_classes} monkey classes: {class_names}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Update device if specified
    if args.device != 'auto':
        config['training']['device'] = args.device

    # Initialize experiment tracker
    experiment_tracker = ExperimentTracker()
    experiment_id = experiment_tracker.start_experiment(
        experiment_name=args.experiment,
        config=config,
        description=f"Training with {num_classes} monkey classes"
    )

    print(f"Started experiment: {experiment_id}")

    try:
        # Initialize trainer
        trainer = RecognitionTrainer(
            train_dir=train_dir,
            val_dir=None,  # No validation directory in this setup
            num_classes=num_classes,
            backbone=config['model']['backbone'],
            embedding_size=config['model']['embedding_size'],
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            device=config['training']['device'],
            experiment_tracker=experiment_tracker
        )

        # Train model
        print("Starting training...")
        model_path = trainer.train(
            epochs=config['training']['epochs'],
            save_dir="models/recognition",
            model_name="monkey_recognition_model"
        )

        print(f"Training completed! Model saved to: {model_path}")

        # Get final metrics
        final_metrics = trainer.get_final_metrics()
        print("Final metrics:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Finish experiment
        experiment_tracker.finish_experiment(
            status="completed",
            final_metrics=final_metrics,
            notes="Training completed successfully"
        )

    except Exception as e:
        print(f"Training failed: {e}")
        experiment_tracker.finish_experiment(
            status="failed",
            notes=f"Training failed: {str(e)}"
        )
        sys.exit(1)


if __name__ == '__main__':
    main()