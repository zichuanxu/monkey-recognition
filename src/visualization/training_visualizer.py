"""
Training Stage Visualization Module for Monkey Face Recognition System

This module provides visualization tools for analyzing the training process,
model architecture, and training data distribution.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class TrainingVisualizer:
    """Comprehensive visualization tool for training analysis."""

    def __init__(self, model_path: str = "models/recognition/monkey_recognition_model_best.pt",
                 data_dir: str = "data/train_Magface",
                 experiment_dir: str = "experiments"):
        """
        Initialize training visualizer.

        Args:
            model_path: Path to the trained model file
            data_dir: Path to training data directory
            experiment_dir: Path to experiments directory
        """
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.experiment_dir = Path(experiment_dir)

        # Load model and extract information
        self.model_info = self._load_model_info()
        self.data_stats = self._analyze_training_data()

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    def _load_model_info(self) -> Dict:
        """Load and analyze model information."""
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')

            model_info = {
                'checkpoint_keys': list(checkpoint.keys()),
                'model_state_dict': checkpoint.get('model_state_dict', {}),
                'optimizer_state_dict': checkpoint.get('optimizer_state_dict', {}),
                'epoch': checkpoint.get('epoch', 'Unknown'),
                'best_accuracy': checkpoint.get('best_accuracy', 'Unknown'),
                'training_config': checkpoint.get('config', {}),
                'model_architecture': self._analyze_model_architecture(checkpoint)
            }

            return model_info

        except Exception as e:
            print(f"Warning: Could not load model file: {e}")
            return {'error': str(e)}

    def _analyze_model_architecture(self, checkpoint: Dict) -> Dict:
        """Analyze model architecture from checkpoint."""
        arch_info = {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'layer_info': [],
            'parameter_distribution': {}
        }

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']

            # Count parameters by layer type
            layer_types = {}
            for name, param in state_dict.items():
                layer_type = name.split('.')[0] if '.' in name else name
                param_count = param.numel() if hasattr(param, 'numel') else 0

                if layer_type not in layer_types:
                    layer_types[layer_type] = 0
                layer_types[layer_type] += param_count
                arch_info['total_parameters'] += param_count

            arch_info['parameter_distribution'] = layer_types
            arch_info['trainable_pameters'] = arch_info['total_parameters']  # Assume all trainable

        return arch_info

    def _analyze_training_data(self) -> Dict:
        """Analyze training data distribution."""
        data_stats = {
            'total_classes': 0,
            'total_images': 0,
            'class_distribution': {},
            'images_per_class': [],
            'class_balance_stats': {}
        }

        if not self.data_dir.exists():
            print(f"Warning: Training data directory not found: {self.data_dir}")
            return data_stats

        try:
            # Count images per class
            class_counts = {}
            for class_dir in self.data_dir.iterdir():
                if class_dir.is_dir():
                    image_count = len([f for f in class_dir.iterdir()
                                     if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                    class_counts[class_dir.name] = image_count

            data_stats['total_classes'] = len(class_counts)
            data_stats['total_images'] = sum(class_counts.values())
            data_stats['class_distribution'] = class_counts
            data_stats['images_per_class'] = list(class_counts.values())

            # Calculate balance statistics
            if class_counts:
                images_per_class = list(class_counts.values())
                data_stats['class_balance_stats'] = {
                    'mean_images_per_class': np.mean(images_per_class),
                    'std_images_per_class': np.std(images_per_class),
                    'min_images_per_class': min(images_per_class),
                    'max_images_per_class': max(images_per_class),
                    'median_images_per_class': np.median(images_per_class)
                }

        except Exception as e:
            print(f"Warning: Could not analyze training data: {e}")

        return data_stats

    def plot_model_architecture(self, save_path: Optional[str] = None) -> None:
        """
        Visualize model architecture and parameter distribution.

        Args:
            save_path: Optional path to save the plot
        """
        if 'error' in self.model_info:
            print(f"Cannot plot architecture: {self.model_info['error']}")
            return

        arch_info = self.model_info['model_architecture']
        param_dist = arch_info['parameter_distribution']

        if not param_dist:
            print("No parameter distribution data available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Model Architecture Analysis', fontsize=16, fontweight='bold')

        # Parameter distribution by layer type
        layers = list(param_dist.keys())
        params = list(param_dist.values())

        # Sort by parameter count
        sorted_data = sorted(zip(layers, params), key=lambda x: x[1], reverse=True)
        layers, params = zip(*sorted_data)

        bars1 = ax1.bar(range(len(layers)), params, color='skyblue', alpha=0.7)
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels(layers, rotation=45, ha='right')
        ax1.set_ylabel('Number of Parameters')
        ax1.set_title('Parameters by Layer Type')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, params):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,}', ha='center', va='bottom', rotation=90)

        # Parameter distribution pie chart
        # Group smaller layers together
        threshold = sum(params) * 0.02  # 2% threshold
        major_layers = []
        major_params = []
        other_params = 0

        for layer, param_count in zip(layers, params):
            if param_count > threshold:
                major_layers.append(layer)
                major_params.append(param_count)
            else:
                other_params += param_count

        if other_params > 0:
            major_layers.append('Others')
            major_params.append(other_params)

        ax2.pie(major_params, labels=major_layers, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Parameter Distribution')

        # Add total parameters info
        total_params = arch_info['total_parameters']
        fig.text(0.5, 0.02, f'Total Parameters: {total_params:,}',
                ha='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_data_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Visualize training data distribution.

        Args:
            save_path: Optional path to save the plot
        """
        if not self.data_stats['class_distribution']:
            print("No training data distribution available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Data Distribution Analysis', fontsize=16, fontweight='bold')

        class_counts = self.data_stats['class_distribution']
        images_per_class = self.data_stats['images_per_class']

        # 1. Class distribution histogram
        ax1 = axes[0, 0]
        ax1.hist(images_per_class, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_xlabel('Images per Class')
        ax1.set_ylabel('Number of Classes')
        ax1.set_title('Distribution of Images per Class')
        ax1.grid(True, alpha=0.3)

        # Add statistics
        stats = self.data_stats['class_balance_stats']
        ax1.axvline(stats['mean_images_per_class'], color='red', linestyle='--',
                   label=f"Mean: {stats['mean_images_per_class']:.1f}")
        ax1.axvline(stats['median_images_per_class'], color='orange', linestyle='--',
                   label=f"Median: {stats['median_images_per_class']:.1f}")
        ax1.legend()

        # 2. Top 10 classes by image count
        ax2 = axes[0, 1]
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        classes, counts = zip(*top_classes)

        bars = ax2.bar(range(len(classes)), counts, color='lightgreen', alpha=0.7)
        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in classes],
                           rotation=45, ha='right')
        ax2.set_ylabel('Number of Images')
        ax2.set_title('Top 10 Classes by Image Count')

        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    str(count), ha='center', va='bottom')

        # 3. Bottom 10 classes by image count
        ax3 = axes[1, 0]
        bottom_classes = sorted(class_counts.items(), key=lambda x: x[1])[:10]
        classes, counts = zip(*bottom_classes)

        bars = ax3.bar(range(len(classes)), counts, color='lightcoral', alpha=0.7)
        ax3.set_xticks(range(len(classes)))
        ax3.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in classes],
                           rotation=45, ha='right')
        ax3.set_ylabel('Number of Images')
        ax3.set_title('Bottom 10 Classes by Image Count')

        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    str(count), ha='center', va='bottom')

        # 4. Data balance analysis
        ax4 = axes[1, 1]

        # Create balance categories
        balance_categories = {
            'Very Low (1-10)': len([c for c in images_per_class if 1 <= c <= 10]),
            'Low (11-50)': len([c for c in images_per_class if 11 <= c <= 50]),
            'Medium (51-200)': len([c for c in images_per_class if 51 <= c <= 200]),
            'High (201-500)': len([c for c in images_per_class if 201 <= c <= 500]),
            'Very High (500+)': len([c for c in images_per_class if c > 500])
        }

        categories = list(balance_categories.keys())
        values = list(balance_categories.values())

        colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99', '#cc99ff']
        ax4.pie(values, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
        ax4.set_title('Class Balance Distribution')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_training_summary(self, save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive training summary visualization.

        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Summary Dashboard', fontsize=18, fontweight='bold')

        # 1. Model Configuration
        ax1 = axes[0, 0]
        ax1.axis('off')

        config_text = "Model Configuration:\n\n"
        if 'training_config' in self.model_info and self.model_info['training_config']:
            config = self.model_info['training_config']
            if 'model' in config:
                model_config = config['model']
                config_text += f"• Backbone: {model_config.get('backbone', 'Unknown')}\n"
                config_text += f"• Embedding Size: {model_config.get('embedding_size', 'Unknown')}\n"
                config_text += f"• Margin Loss: {model_config.get('margin_loss', 'Unknown')}\n"
                config_text += f"• Pretrained: {model_config.get('pretrained', 'Unknown')}\n"

            if 'training' in config:
                train_config = config['training']
                config_text += f"\nTraining Config:\n"
                config_text += f"• Batch Size: {train_config.get('batch_size', 'Unknown')}\n"
                config_text += f"• Learning Rate: {train_config.get('learning_rate', 'Unknown')}\n"
                config_text += f"• Epochs: {train_config.get('epochs', 'Unknown')}\n"
        else:
            config_text += "Configuration not available"

        ax1.text(0.05, 0.95, config_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax1.set_title('Model Configuration', fontweight='bold')

        # 2. Dataset Statistics
        ax2 = axes[0, 1]
        ax2.axis('off')

        stats_text = "Dataset Statistics:\n\n"
        stats_text += f"• Total Classes: {self.data_stats['total_classes']}\n"
        stats_text += f"• Total Images: {self.data_stats['total_images']:,}\n"

        if self.data_stats['class_balance_stats']:
            balance_stats = self.data_stats['class_balance_stats']
            stats_text += f"• Avg Images/Class: {balance_stats['mean_images_per_class']:.1f}\n"
            stats_text += f"• Min Images/Class: {balance_stats['min_images_per_class']}\n"
            stats_text += f"• Max Images/Class: {balance_stats['max_images_per_class']}\n"
            stats_text += f"• Std Dev: {balance_stats['std_images_per_class']:.1f}\n"

        # Calculate data balance ratio
        if self.data_stats['class_balance_stats']:
            min_imgs = balance_stats['min_images_per_class']
            max_imgs = balance_stats['max_images_per_class']
            balance_ratio = max_imgs / min_imgs if min_imgs > 0 else float('inf')
            stats_text += f"• Balance Ratio: {balance_ratio:.1f}:1\n"

        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        ax2.set_title('Dataset Statistics', fontweight='bold')

        # 3. Model Size Analysis
        ax3 = axes[0, 2]
        if 'model_architecture' in self.model_info:
            arch_info = self.model_info['model_architecture']
            total_params = arch_info['total_parameters']

            # Estimate model size in MB (assuming float32)
            model_size_mb = (total_params * 4) / (1024 * 1024)

            sizes = ['Parameters', 'Model Size (MB)']
            values = [total_params / 1e6, model_size_mb]  # Convert params to millions

            bars = ax3.bar(sizes, values, color=['lightgreen', 'lightcoral'], alpha=0.7)
            ax3.set_ylabel('Count')
            ax3.set_title('Model Size Analysis')

            # Add value labels
            ax3.text(0, values[0] + values[0]*0.05, f'{total_params/1e6:.1f}M',
                    ha='center', va='bottom', fontweight='bold')
            ax3.text(1, values[1] + values[1]*0.05, f'{model_size_mb:.1f}MB',
                    ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Model info\nnot available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Model Size Analysis')

        # 4. Class Distribution Overview
        ax4 = axes[1, 0]
        if self.data_stats['images_per_class']:
            images_per_class = self.data_stats['images_per_class']
            ax4.hist(images_per_class, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.set_xlabel('Images per Class')
            ax4.set_ylabel('Number of Classes')
            ax4.set_title('Class Size Distribution')
            ax4.grid(True, alpha=0.3)

        # 5. Data Imbalance Visualization
        ax5 = axes[1, 1]
        if self.data_stats['images_per_class']:
            images_per_class = self.data_stats['images_per_class']

            # Create imbalance categories
            categories = ['1-10', '11-50', '51-200', '201-500', '500+']
            counts = [
                len([c for c in images_per_class if 1 <= c <= 10]),
                len([c for c in images_per_class if 11 <= c <= 50]),
                len([c for c in images_per_class if 51 <= c <= 200]),
                len([c for c in images_per_class if 201 <= c <= 500]),
                len([c for c in images_per_class if c > 500])
            ]

            colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
            bars = ax5.bar(categories, counts, color=colors, alpha=0.7)
            ax5.set_xlabel('Images per Class Range')
            ax5.set_ylabel('Number of Classes')
            ax5.set_title('Data Balance Analysis')

            # Add value labels
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            str(count), ha='center', va='bottom')

        # 6. Training Progress (if available)
        ax6 = axes[1, 2]
        ax6.axis('off')

        progress_text = "Training Information:\n\n"
        if 'epoch' in self.model_info:
            progress_text += f"• Final Epoch: {self.model_info['epoch']}\n"
        if 'best_accuracy' in self.model_info:
            progress_text += f"• Best Accuracy: {self.model_info['best_accuracy']}\n"

        # Add timestamp info if available
        progress_text += f"\nModel File:\n• {Path(self.model_path).name}\n"

        ax6.text(0.05, 0.95, progress_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        ax6.set_title('Training Progress', fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_training_report(self) -> str:
        """Generate a comprehensive text report of training analysis."""
        report = f"""
=== MONKEY RECOGNITION MODEL - TRAINING ANALYSIS REPORT ===

Model Information:
- Model File: {self.model_path}
- Model Status: {'Loaded Successfully' if 'error' not in self.model_info else 'Error Loading'}
"""

        if 'error' not in self.model_info:
            arch_info = self.model_info['model_architecture']
            report += f"- Total Parameters: {arch_info['total_parameters']:,}\n"
            report += f"- Model Size: {(arch_info['total_parameters'] * 4) / (1024 * 1024):.1f} MB\n"

            if 'epoch' in self.model_info:
                report += f"- Training Epochs: {self.model_info['epoch']}\n"
            if 'best_accuracy' in self.model_info:
                report += f"- Best Accuracy: {self.model_info['best_accuracy']}\n"

        report += f"""
Dataset Analysis:
- Training Data Directory: {self.data_dir}
- Total Classes: {self.data_stats['total_classes']}
- Total Training Images: {self.data_stats['total_images']:,}
"""

        if self.data_stats['class_balance_stats']:
            stats = self.data_stats['class_balance_stats']
            report += f"- Average Images per Class: {stats['mean_images_per_class']:.1f}\n"
            report += f"- Standard Deviation: {stats['std_images_per_class']:.1f}\n"
            report += f"- Min Images per Class: {stats['min_images_per_class']}\n"
            report += f"- Max Images per Class: {stats['max_images_per_class']}\n"

            balance_ratio = stats['max_images_per_class'] / stats['min_images_per_class'] if stats['min_images_per_class'] > 0 else float('inf')
            report += f"- Data Imbalance Ratio: {balance_ratio:.1f}:1\n"

        # Data balance analysis
        if self.data_stats['images_per_class']:
            images_per_class = self.data_stats['images_per_class']
            report += f"""
Data Balance Distribution:
- Very Low (1-10 images): {len([c for c in images_per_class if 1 <= c <= 10])} classes
- Low (11-50 images): {len([c for c in images_per_class if 11 <= c <= 50])} classes
- Medium (51-200 images): {len([c for c in images_per_class if 51 <= c <= 200])} classes
- High (201-500 images): {len([c for c in images_per_class if 201 <= c <= 500])} classes
- Very High (500+ images): {len([c for c in images_per_class if c > 500])} classes
"""

        return report

    def create_comprehensive_training_report(self, output_dir: str = "training_analysis") -> None:
        """
        Create a comprehensive training analysis report with all visualizations.

        Args:
            output_dir: Directory to save all visualizations
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        print("Generating comprehensive training analysis report...")

        # Generate all plots
        self.plot_model_architecture(f"{output_dir}/01_model_architecture.png")
        self.plot_data_distribution(f"{output_dir}/02_data_distribution.png")
        self.plot_training_summary(f"{output_dir}/03_training_summary.png")

        # Save text report
        report = self.generate_training_report()
        with open(f"{output_dir}/training_analysis_report.txt", 'w') as f:
            f.write(report)

        print(f"Training analysis report generated in: {output_dir}/")
        print("\nGenerated files:")
        print("- 01_model_architecture.png")
        print("- 02_data_distribution.png")
        print("- 03_training_summary.png")
        print("- training_analysis_report.txt")

        # Print summary to console
        print(report)


def main():
    """Example usage of the TrainingVisualizer."""
    # Initialize visualizer
    visualizer = TrainingVisualizer()

    # Generate comprehensive report
    visualizer.create_comprehensive_training_report()


if __name__ == "__main__":
    main()