#!/usr/bin/env python3
"""
Training visualization script for the monkey recognition model.
"""

from src.visualization.training_visualizer import TrainingVisualizer

def main():
    """Run training stage visualizations."""
    print("🐒 Monkey Recognition Model - Training Analysis")
    print("=" * 55)

    try:
        # Initialize visualizer
        visualizer = TrainingVisualizer(
            model_path="models/recognition/monkey_recognition_model_best.pt",
            data_dir="data/train_Magface"
        )

        # Create comprehensive training analysis report
        visualizer.create_comprehensive_training_report("training_analysis")

        print("\n✅ Training analysis complete! Check the 'training_analysis' folder.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the model file and training data directory exist.")

if __name__ == "__main__":
    main()