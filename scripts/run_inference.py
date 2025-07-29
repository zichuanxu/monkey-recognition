#!/usr/bin/env python3
"""Simple inference script for monkey face recognition system."""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.logging import setup_logging
from src.utils.image_utils import load_image


def run_inference_on_image(image_path: str, output_dir: str = None) -> dict:
    """Run inference on a single image.

    Args:
        image_path: Path to input image.
        output_dir: Optional output directory for results.

    Returns:
        Dictionary with inference results.
    """
    print(f"Processing image: {image_path}")

    # Load image
    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return {"error": "Failed to load image"}

    # Placeholder for actual inference
    # In a complete implementation, this would use the trained model
    result = {
        "image_path": image_path,
        "image_shape": image.shape,
        "status": "processed",
        "note": "This is a placeholder. Actual inference requires trained models."
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        image_name = Path(image_path).stem
        results_path = output_dir / f"{image_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {results_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run inference on monkey images")

    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to input image'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='inference_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level='INFO')

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    try:
        # Run inference
        result = run_inference_on_image(args.image, args.output)

        print("\\nInference completed!")
        print(f"Image: {result.get('image_path', 'N/A')}")
        print(f"Shape: {result.get('image_shape', 'N/A')}")
        print(f"Status: {result.get('status', 'N/A')}")

        if 'note' in result:
            print(f"Note: {result['note']}")

    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()