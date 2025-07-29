#!/usr/bin/env python3
"""Example usage of the monkey face recognition system."""

import os
import sys

def main():
    print("Monkey Face Recognition System - Example Usage")
    print("=" * 50)

    print("\\n1. Setup (run once):")
    print("   python setup.py")

    print("\\n2. Data preparation:")
    print("   - Place training images in: data/train_Magface/monkey_id/")
    print("   - Place test images in: test_image/monkey_id/")
    print("   - Example structure:")
    print("     data/train_Magface/")
    print("     ├── monkey_001/")
    print("     │   ├── img_001.jpg")
    print("     │   └── img_002.jpg")
    print("     └── monkey_002/")
    print("         └── img_003.jpg")

    print("\\n3. Training:")
    print("   python scripts/train_model.py --config config/training_config.yaml")

    print("\\n4. Evaluation:")
    print("   python scripts/evaluate_model.py --config config/training_config.yaml")

    print("\\n5. Inference:")
    print("   python scripts/run_inference.py --image path/to/image.jpg")

    print("\\n" + "=" * 50)
    print("For more information, see README.md")

if __name__ == '__main__':
    main()