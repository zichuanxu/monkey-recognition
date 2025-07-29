#!/usr/bin/env python3
"""Setup script for monkey face recognition system."""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/train_Magface",
        "data/test_image",
        "models/recognition",
        "models/databases",
        "experiments",
        "logs",
        "results"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'opencv-python',
        'PIL', 'yaml', 'sklearn', 'matplotlib', 'seaborn', 'pandas'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'PIL':
                import PIL
            elif package == 'yaml':
                import yaml
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")

    if missing_packages:
        print(f"\\nMissing packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False

    return True

def main():
    print("Setting up Monkey Face Recognition System...")
    print("=" * 50)

    # Create directories
    print("\\n1. Creating directories...")
    create_directories()

    # Check dependencies
    print("\\n2. Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)

    # Check CUDA
    print("\\n3. Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available (CPU-only mode)")
    except ImportError:
        print("✗ PyTorch not installed")

    print("\\n" + "=" * 50)
    print("Setup completed!")
    print("\\nNext steps:")
    print("1. Place your training data in: data/train_Magface/")
    print("2. Place your test data in: test_image/")
    print("3. Run training: python scripts/train_model.py")
    print("4. Run evaluation: python scripts/evaluate_model.py")

if __name__ == '__main__':
    main()