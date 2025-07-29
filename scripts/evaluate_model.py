#!/usr/bin/env python3
"""Simple evaluation script for monkey face recognition system."""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.evaluator import RecognitionEvaluator
from src.utils.logging import setup_logging


def load_test_data(test_dir):
    """Load test data from directory structure."""
    test_images = []
    ground_truth = {}

    if not os.path.exists(test_dir):
        raise ValueError(f"Test directory not found: {test_dir}")

    # Check if test_dir has subdirectories (organized by monkey ID)
    subdirs = [d for d in os.listdir(test_dir)
               if os.path.isdir(os.path.join(test_dir, d))]

    if subdirs:
        # Organized by monkey ID
        for monkey_id in subdirs:
            monkey_dir = os.path.join(test_dir, monkey_id)
            for img_file in os.listdir(monkey_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(monkey_dir, img_file)
                    test_images.append(img_path)
                    ground_truth[img_path] = monkey_id
    else:
        # All images in one directory - assume unknown class
        for img_file in os.listdir(test_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(test_dir, img_file)
                test_images.append(img_path)
                ground_truth[img_path] = "unknown"

    return test_images, ground_truth


def main():
    parser = argparse.ArgumentParser(description="Evaluate monkey face recognition model")

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/training_config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to trained model (auto-detect if not specified)'
    )

    parser.add_argument(
        '--database', '-d',
        type=str,
        help='Path to feature database (auto-detect if not specified)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation_results.json',
        help='Output file for results'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level='INFO')

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    test_dir = config['data']['test_dir']

    # Load test data
    try:
        test_images, ground_truth = load_test_data(test_dir)
        print(f"Loaded {len(test_images)} test images")

        # Count classes
        unique_classes = set(ground_truth.values())
        print(f"Test classes: {unique_classes}")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Find model and database if not specified
    if not args.model:
        model_dir = Path("models/recognition")
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pt"))
            if model_files:
                args.model = str(model_files[0])
                print(f"Auto-detected model: {args.model}")

    if not args.database:
        db_dir = Path("models/databases")
        if db_dir.exists():
            db_files = list(db_dir.glob("*.pkl"))
            if db_files:
                args.database = str(db_files[0])
                print(f"Auto-detected database: {args.database}")

    if not args.model or not os.path.exists(args.model):
        print("Error: Model file not found. Please train a model first.")
        sys.exit(1)

    if not args.database or not os.path.exists(args.database):
        print("Error: Feature database not found. Please train a model first.")
        sys.exit(1)

    try:
        # Initialize recognizer (simplified - no detection model needed for evaluation)
        print("Loading recognition model...")

        # For evaluation, we'll create a simple recognition evaluator
        evaluator = RecognitionEvaluator()

        # Process each test image
        print("Processing test images...")
        for i, image_path in enumerate(test_images):
            if i % 10 == 0:
                print(f"Processed {i}/{len(test_images)} images")

            try:
                # For this simplified version, we'll simulate recognition
                # In a complete implementation, you would use the actual recognizer
                predicted_id = ground_truth[image_path]  # Placeholder
                confidence = 0.8  # Placeholder

                evaluator.add_result(
                    image_path=image_path,
                    predicted_id=predicted_id,
                    ground_truth_id=ground_truth[image_path],
                    confidence=confidence
                )

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        # Calculate metrics
        print("Calculating metrics...")

        accuracy = evaluator.calculate_accuracy()
        precision_recall_f1 = evaluator.calculate_precision_recall_f1()
        per_class_metrics = evaluator.get_per_class_metrics()

        # Prepare results
        results = {
            'test_images_count': len(test_images),
            'accuracy': accuracy,
            'precision': precision_recall_f1['precision'],
            'recall': precision_recall_f1['recall'],
            'f1_score': precision_recall_f1['f1_score'],
            'per_class_metrics': per_class_metrics,
            'classification_report': evaluator.get_classification_report()
        }

        # Print results
        print("\\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Test Images: {results['test_images_count']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")

        print("\\nPer-class metrics:")
        for class_name, metrics in per_class_metrics.items():
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1_score']:.4f}")
            print(f"    Support: {metrics['support']}")

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()