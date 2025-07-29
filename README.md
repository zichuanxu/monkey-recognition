# Monkey Face Recognition System

A deep learning system for recognizing individual monkeys in images using ArcFace-based recognition.

## Overview

This system uses deep learning to identify individual monkeys from facial features. It's designed for wildlife research and conservation efforts where individual identification is crucial.

## Quick Start

### Get Data

Get the data through the dropbox link on CLE:

```
Constructing a monkey identification model
• Training data + Test data
• Dropbox(25GB):
```

Then create a dir called `data` and put them in the  `data` dir.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/zichuanxu/monkey-recognition.git
   cd monkey-recognition
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Data Structure

Organize your data as follows:

```
data/
└── train_Magface/
    ├── monkey_001/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    ├── monkey_002/
    │   └── ...
    └── ...
└──test_image/
    ├── monkey_001/
    │   └── test_images...
    ├── monkey_002/
    │   └── test_images...
    └── ...

```

### Training

1. **Configure training:**

   ```bash
   # Edit config/training_config.yaml if needed
   ```
2. **Train the model:**

   ```bash
   python scripts/train_model.py --config config/training_config.yaml --experiment "my_experiment"
   ```

### Evaluation

```bash
python scripts/evaluate_model.py --config config/training_config.yaml --output evaluation_results.json
```

### Inference

```bash
python scripts/run_inference.py --image path/to/image.jpg --output results/
```

## Project Structure

```
monkey-recognition/
├── data/  		          # Data (got by the link from the teacher)
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   ├── recognition/              # Face recognition components
│   ├── training/                 # Training orchestration
│   ├── evaluation/               # Evaluation framework
│   ├── inference/                # Inference pipeline
│   ├── visualization/            # Visualization tools
│   └── utils/                    # Utility functions
├── scripts/                      # Executable scripts
│   ├── train_model.py           # Training script
│   ├── evaluate_model.py        # Evaluation script
│   └── run_inference.py         # Inference script
├── config/                       # Configuration files
│   └── training_config.yaml     # Training configuration
├── models/                       # Trained models (created after training)
├── experiments/                  # Experiment tracking (created after training)
└── requirements.txt              # Dependencies
```

## Configuration

The system uses `config/training_config.yaml` for configuration:

```yaml
# Data configuration
data:
  train_dir: "data/train_Magface"   # Training data directory
  test_dir: "data/test_image"       # Test data directory
  image_size: 224                   # Input image size

# Training configuration
training:
  device: "auto"                    # Device: auto, cpu, cuda
  batch_size: 32                    # Batch size
  learning_rate: 0.001              # Learning rate
  epochs: 100                       # Training epochs

# Model configuration
model:
  backbone: "resnet50"              # Recognition backbone
  embedding_size: 512               # Feature embedding size
  margin_loss: "arcface"            # Margin loss type
```

## Training Process

1. **Data Preparation**: Organize images by monkey ID in `/data/train_Magface/`
2. **Model Training**: Run training script to train ArcFace recognition model
3. **Feature Database**: System creates feature database for known monkeys
4. **Evaluation**: Test model performance on `/data/test_image/` directory

## Performance Metrics

The system provides:

- **Accuracy**: Overall recognition accuracy
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed per-class analysis
