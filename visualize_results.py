#!/usr/bin/env python3
"""
Quick script to visualize test results for the monkey recognition model.
"""

from src.visualization.test_results_visualizer import TestResultsVisualizer

def main():
    """Run visualization of test results."""
    print("🐒 Monkey Recognition Model - Test Results Visualization")
    print("=" * 60)

    try:
        # Initialize visualizer
        visualizer = TestResultsVisualizer("evaluation_results.json")

        # Create comprehensive report with all visualizations
        visualizer.create_comprehensive_report("results_visualization")

        print("\n✅ Visualization complete! Check the 'results_visualization' folder.")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure 'evaluation_results.json' exists in the current directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()