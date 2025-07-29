"""Experiment tracking and model management for monkey recognition training."""

import os
import json
import pickle
import shutil
import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import torch
import numpy as np

from ..utils.logging import LoggerMixin
from ..utils.error_handler import handle_errors
from ..utils.exceptions import TrainingError, ConfigurationError, FileSystemError


class ExperimentTracker(LoggerMixin):
    """Track training experiments and manage model versions."""

    def __init__(self, experiments_dir: str = "experiments"):
        """Initialize experiment tracker.

        Args:
            experiments_dir: Directory to store experiment data.
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)

        # Current experiment info
        self.current_experiment = None
        self.current_run = None

        self.logger.info(f"ExperimentTracker initialized with directory: {experiments_dir}")

    @handle_errors(reraise=True)
    def start_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """Start a new experiment.

        Args:
            experiment_name: Name of the experiment.
            config: Experiment configuration.
            description: Optional description.
            tags: Optional tags for categorization.

        Returns:
            Experiment ID.
        """
        # Create experiment directory
        experiment_dir = self.experiments_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)

        # Generate run ID with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        run_dir = experiment_dir / run_id
        run_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "metrics").mkdir(exist_ok=True)
        (run_dir / "visualizations").mkdir(exist_ok=True)

        # Save experiment metadata
        metadata = {
            "experiment_name": experiment_name,
            "run_id": run_id,
            "start_time": datetime.datetime.now().isoformat(),
            "config": config,
            "description": description,
            "tags": tags or [],
            "status": "running",
            "metrics": {},
            "artifacts": []
        }

        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update current experiment
        self.current_experiment = experiment_name
        self.current_run = run_id

        self.logger.info(f"Started experiment '{experiment_name}' with run ID '{run_id}'")
        return f"{experiment_name}/{run_id}"

    @handle_errors(reraise=False)
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ) -> None:
        """Log training metrics.

        Args:
            metrics: Dictionary of metric values.
            step: Optional training step.
            epoch: Optional training epoch.
        """
        if not self.current_experiment or not self.current_run:
            self.logger.warning("No active experiment to log metrics to")
            return

        run_dir = self.experiments_dir / self.current_experiment / self.current_run
        metrics_file = run_dir / "metrics" / "training_metrics.jsonl"

        # Create metrics entry
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics,
            "step": step,
            "epoch": epoch
        }

        # Append to metrics file
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # Update metadata
        self._update_metadata({"last_metrics": metrics})

    @handle_errors(reraise=True)
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save model checkpoint.

        Args:\n            model: PyTorch model to save.
            optimizer: Optimizer state to save.
            epoch: Current epoch.
            metrics: Current metrics.
            is_best: Whether this is the best checkpoint.
            additional_data: Additional data to save.

        Returns:
            Path to saved checkpoint.
        """
        if not self.current_experiment or not self.current_run:
            raise TrainingError(
                "No active experiment to save checkpoint to",
                error_code="NO_ACTIVE_EXPERIMENT"
            )

        run_dir = self.experiments_dir / self.current_experiment / self.current_run
        checkpoints_dir = run_dir / "checkpoints"

        # Create checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.datetime.now().isoformat()
        }

        if additional_data:
            checkpoint_data.update(additional_data)

        # Save checkpoint
        checkpoint_filename = f"checkpoint_epoch_{epoch:04d}.pt"
        checkpoint_path = checkpoints_dir / checkpoint_filename

        torch.save(checkpoint_data, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = checkpoints_dir / "best_checkpoint.pt"
            shutil.copy2(checkpoint_path, best_path)
            self.logger.info(f"Saved best checkpoint at epoch {epoch}")

        # Update metadata
        self._update_metadata({
            "last_checkpoint": str(checkpoint_path),
            "last_epoch": epoch,
            "last_metrics": metrics
        })

        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        return str(checkpoint_path)

    @handle_errors(reraise=True)
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            model: Model to load state into.
            optimizer: Optional optimizer to load state into.
            device: Device to load checkpoint on.

        Returns:
            Checkpoint metadata.
        """
        if not os.path.exists(checkpoint_path):
            raise FileSystemError(
                f"Checkpoint not found: {checkpoint_path}",
                error_code="CHECKPOINT_NOT_FOUND"
            )

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load model state
            model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer state if provided
            if optimizer and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Return metadata
            metadata = {
                "epoch": checkpoint.get("epoch", 0),
                "metrics": checkpoint.get("metrics", {}),
                "timestamp": checkpoint.get("timestamp", ""),
                "checkpoint_path": checkpoint_path
            }

            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return metadata

        except Exception as e:
            raise TrainingError(
                f"Failed to load checkpoint: {e}",
                error_code="CHECKPOINT_LOAD_FAILED",
                details={"checkpoint_path": checkpoint_path}
            )

    @handle_errors(reraise=False)
    def save_artifact(
        self,
        artifact_data: Any,
        artifact_name: str,
        artifact_type: str = "pickle"
    ) -> Optional[str]:
        """Save training artifact.

        Args:
            artifact_data: Data to save.
            artifact_name: Name of the artifact.
            artifact_type: Type of artifact (pickle, json, numpy).

        Returns:
            Path to saved artifact or None if failed.
        """
        if not self.current_experiment or not self.current_run:
            self.logger.warning("No active experiment to save artifact to")
            return None

        run_dir = self.experiments_dir / self.current_experiment / self.current_run

        try:
            if artifact_type == "pickle":
                artifact_path = run_dir / f"{artifact_name}.pkl"
                with open(artifact_path, 'wb') as f:
                    pickle.dump(artifact_data, f)
            elif artifact_type == "json":
                artifact_path = run_dir / f"{artifact_name}.json"
                with open(artifact_path, 'w') as f:
                    json.dump(artifact_data, f, indent=2)
            elif artifact_type == "numpy":
                artifact_path = run_dir / f"{artifact_name}.npy"
                np.save(artifact_path, artifact_data)
            else:
                self.logger.error(f"Unsupported artifact type: {artifact_type}")
                return None

            # Update metadata
            self._update_metadata({
                "artifacts": self._get_current_artifacts() + [str(artifact_path)]
            })

            self.logger.info(f"Saved artifact: {artifact_path}")
            return str(artifact_path)

        except Exception as e:
            self.logger.error(f"Failed to save artifact {artifact_name}: {e}")
            return None

    @handle_errors(reraise=False)
    def finish_experiment(
        self,
        status: str = "completed",
        final_metrics: Optional[Dict[str, float]] = None,
        notes: str = ""
    ) -> None:
        """Finish current experiment.

        Args:
            status: Final status (completed, failed, stopped).
            final_metrics: Final metrics summary.
            notes: Optional notes about the experiment.
        """
        if not self.current_experiment or not self.current_run:
            self.logger.warning("No active experiment to finish")
            return

        # Update metadata
        updates = {
            "status": status,
            "end_time": datetime.datetime.now().isoformat(),
            "notes": notes
        }

        if final_metrics:
            updates["final_metrics"] = final_metrics

        self._update_metadata(updates)

        self.logger.info(
            f"Finished experiment '{self.current_experiment}/{self.current_run}' with status: {status}"
        )

        # Clear current experiment
        self.current_experiment = None
        self.current_run = None

    def get_experiment_info(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment information.

        Args:
            experiment_id: Experiment ID in format 'experiment_name/run_id'.

        Returns:
            Experiment metadata or None if not found.
        """
        try:
            experiment_name, run_id = experiment_id.split('/')
            metadata_path = self.experiments_dir / experiment_name / run_id / "metadata.json"

            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Failed to get experiment info for {experiment_id}: {e}")
            return None

    def list_experiments(self, experiment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments.

        Args:
            experiment_name: Optional filter by experiment name.

        Returns:
            List of experiment metadata.
        """
        experiments = []

        try:
            if experiment_name:
                experiment_dirs = [self.experiments_dir / experiment_name]
            else:
                experiment_dirs = [d for d in self.experiments_dir.iterdir() if d.is_dir()]

            for exp_dir in experiment_dirs:
                if not exp_dir.exists():
                    continue

                for run_dir in exp_dir.iterdir():
                    if run_dir.is_dir():
                        metadata_path = run_dir / "metadata.json"
                        if metadata_path.exists():
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                experiments.append(metadata)

            # Sort by start time
            experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)
            return experiments

        except Exception as e:
            self.logger.error(f"Failed to list experiments: {e}")
            return []

    def get_best_checkpoint(self, experiment_id: str) -> Optional[str]:
        """Get path to best checkpoint for an experiment.

        Args:
            experiment_id: Experiment ID in format 'experiment_name/run_id'.

        Returns:
            Path to best checkpoint or None if not found.
        """
        try:
            experiment_name, run_id = experiment_id.split('/')
            best_checkpoint = (
                self.experiments_dir / experiment_name / run_id /
                "checkpoints" / "best_checkpoint.pt"
            )

            if best_checkpoint.exists():
                return str(best_checkpoint)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Failed to get best checkpoint for {experiment_id}: {e}")
            return None

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare.
            metrics: List of metric names to compare.

        Returns:
            Comparison results.
        """
        comparison = {}

        for exp_id in experiment_ids:
            exp_info = self.get_experiment_info(exp_id)
            if exp_info:
                exp_metrics = exp_info.get("final_metrics", {})
                comparison[exp_id] = {
                    "status": exp_info.get("status", "unknown"),
                    "start_time": exp_info.get("start_time", ""),
                    "end_time": exp_info.get("end_time", ""),
                    "config": exp_info.get("config", {}),
                    "metrics": {metric: exp_metrics.get(metric, None) for metric in metrics}
                }

        return comparison

    def _update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update experiment metadata.

        Args:
            updates: Dictionary of updates to apply.
        """
        if not self.current_experiment or not self.current_run:
            return

        run_dir = self.experiments_dir / self.current_experiment / self.current_run
        metadata_path = run_dir / "metadata.json"

        try:
            # Load current metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Apply updates
            metadata.update(updates)

            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to update metadata: {e}")

    def _get_current_artifacts(self) -> List[str]:
        """Get list of current artifacts.

        Returns:
            List of artifact paths.
        """
        if not self.current_experiment or not self.current_run:
            return []

        run_dir = self.experiments_dir / self.current_experiment / self.current_run
        metadata_path = run_dir / "metadata.json"

        try:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get("artifacts", [])
            else:
                return []
        except Exception:
            return []


class ModelManager(LoggerMixin):
    """Manage trained models and their versions."""

    def __init__(self, models_dir: str = "models"):
        """Initialize model manager.

        Args:
            models_dir: Directory to store model files.
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.models_dir / "detection").mkdir(exist_ok=True)
        (self.models_dir / "recognition").mkdir(exist_ok=True)
        (self.models_dir / "databases").mkdir(exist_ok=True)

        self.logger.info(f"ModelManager initialized with directory: {models_dir}")

    @handle_errors(reraise=True)
    def register_model(
        self,
        model_path: str,
        model_type: str,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        experiment_id: Optional[str] = None,
        description: str = ""
    ) -> str:
        """Register a trained model.

        Args:
            model_path: Path to model file.
            model_type: Type of model (detection, recognition).
            model_name: Name of the model.
            version: Model version.
            metrics: Model performance metrics.
            config: Model configuration.
            experiment_id: Optional experiment ID that produced this model.
            description: Optional description.

        Returns:
            Model ID.
        """
        if model_type not in ["detection", "recognition"]:
            raise ConfigurationError(
                f"Invalid model type: {model_type}. Must be 'detection' or 'recognition'",
                error_code="INVALID_MODEL_TYPE"
            )

        # Create model directory
        model_dir = self.models_dir / model_type / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        model_filename = f"{model_name}_v{version}.pt"
        dest_path = model_dir / model_filename
        shutil.copy2(model_path, dest_path)

        # Create model metadata
        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "version": version,
            "model_path": str(dest_path),
            "metrics": metrics,
            "config": config,
            "experiment_id": experiment_id,
            "description": description,
            "registration_time": datetime.datetime.now().isoformat(),
            "file_size": os.path.getsize(dest_path)
        }

        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        model_id = f"{model_type}/{model_name}/{version}"
        self.logger.info(f"Registered model: {model_id}")

        return model_id

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information.

        Args:
            model_id: Model ID in format 'type/name/version'.

        Returns:
            Model metadata or None if not found.
        """
        try:
            model_type, model_name, version = model_id.split('/')
            metadata_path = (
                self.models_dir / model_type / model_name / version / "metadata.json"
            )

            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Failed to get model info for {model_id}: {e}")
            return None

    def list_models(
        self,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List registered models.

        Args:
            model_type: Optional filter by model type.
            model_name: Optional filter by model name.

        Returns:
            List of model metadata.
        """
        models = []

        try:
            if model_type:
                type_dirs = [self.models_dir / model_type]
            else:
                type_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]

            for type_dir in type_dirs:
                if not type_dir.exists():
                    continue

                if model_name:
                    name_dirs = [type_dir / model_name]
                else:
                    name_dirs = [d for d in type_dir.iterdir() if d.is_dir()]

                for name_dir in name_dirs:
                    if not name_dir.exists():
                        continue

                    for version_dir in name_dir.iterdir():
                        if version_dir.is_dir():
                            metadata_path = version_dir / "metadata.json"
                            if metadata_path.exists():
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                                    models.append(metadata)

            # Sort by registration time
            models.sort(key=lambda x: x.get("registration_time", ""), reverse=True)
            return models

        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []

    def get_best_model(
        self,
        model_type: str,
        metric: str,
        higher_is_better: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get best model based on a metric.

        Args:
            model_type: Type of model to search.
            metric: Metric to compare.
            higher_is_better: Whether higher metric values are better.

        Returns:
            Best model metadata or None if not found.
        """
        models = self.list_models(model_type=model_type)

        if not models:
            return None

        # Filter models that have the metric
        valid_models = [m for m in models if metric in m.get("metrics", {})]

        if not valid_models:
            return None

        # Find best model
        if higher_is_better:
            best_model = max(valid_models, key=lambda x: x["metrics"][metric])
        else:
            best_model = min(valid_models, key=lambda x: x["metrics"][metric])

        return best_model

    def get_model_path(self, model_id: str) -> Optional[str]:
        """Get path to model file.

        Args:
            model_id: Model ID in format 'type/name/version'.

        Returns:
            Path to model file or None if not found.
        """
        model_info = self.get_model_info(model_id)
        if model_info:
            model_path = model_info.get("model_path")
            if model_path and os.path.exists(model_path):
                return model_path

        return None

    def delete_model(self, model_id: str) -> bool:
        """Delete a registered model.

        Args:
            model_id: Model ID in format 'type/name/version'.

        Returns:
            True if successful, False otherwise.
        """
        try:
            model_type, model_name, version = model_id.split('/')
            model_dir = self.models_dir / model_type / model_name / version

            if model_dir.exists():
                shutil.rmtree(model_dir)
                self.logger.info(f"Deleted model: {model_id}")
                return True
            else:
                self.logger.warning(f"Model not found: {model_id}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            return False


# Global instances
_experiment_tracker = None
_model_manager = None


def get_experiment_tracker(experiments_dir: str = "experiments") -> ExperimentTracker:
    """Get global experiment tracker instance."""
    global _experiment_tracker
    if _experiment_tracker is None:
        _experiment_tracker = ExperimentTracker(experiments_dir)
    return _experiment_tracker


def get_model_manager(models_dir: str = "models") -> ModelManager:
    """Get global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(models_dir)
    return _model_manager