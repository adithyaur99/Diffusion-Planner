"""
MLflow Logger for Databricks - drop-in replacement for tb_log.py

Usage:
    # In train_predictor.py, change:
    # from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger
    # to:
    # from diffusion_planner.utils.mlflow_log import MLflowLogger as Logger
"""
import os
import json
import mlflow
from typing import Optional


class MLflowLogger:
    """
    MLflow-based logger for Databricks integration.
    Matches the interface of TensorBoardLogger for drop-in replacement.
    """

    def __init__(
        self,
        run_name: str,
        notes: str,
        args,
        wandb_resume_id: Optional[str] = None,  # Kept for interface compatibility
        save_path: Optional[str] = None,
        rank: int = 0
    ):
        """
        Initialize MLflow logging.

        Args:
            run_name: Name for the MLflow run
            notes: Description/notes for the run
            args: Training arguments (argparse namespace)
            wandb_resume_id: Resume ID (mapped to MLflow run_id for compatibility)
            save_path: Path to save artifacts locally
            rank: Process rank (only rank 0 logs to avoid duplicates)
        """
        self.args = args
        self.rank = rank
        self.id = None
        self.save_path = save_path

        if rank != 0:
            return

        # Set experiment name
        experiment_name = f"/Diffusion-Planner/{run_name}"
        mlflow.set_experiment(experiment_name)

        # Resume existing run or start new one
        # wandb_resume_id is mapped to MLflow run_id for compatibility
        if wandb_resume_id:
            try:
                self.run = mlflow.start_run(run_id=wandb_resume_id)
                self.id = wandb_resume_id
                print(f"Resumed MLflow run: {self.id}")
            except Exception as e:
                print(f"Could not resume run {wandb_resume_id}, starting new run: {e}")
                self.run = mlflow.start_run(run_name=run_name)
                self.id = self.run.info.run_id
        else:
            self.run = mlflow.start_run(run_name=run_name)
            self.id = self.run.info.run_id

        # Log notes as tag
        if notes:
            mlflow.set_tag("mlflow.note.content", notes)

        # Log parameters from args
        self._log_params(args)

        # Save args.json artifact if save_path provided
        if save_path:
            self._save_args_artifact(args, save_path)

    def _log_params(self, args):
        """Log training parameters to MLflow."""
        try:
            # Convert args to dict, handling special types
            params_dict = {}
            for k, v in vars(args).items():
                if k.startswith('_'):
                    continue
                # Handle special types that can't be serialized directly
                if hasattr(v, 'to_dict'):
                    # Skip complex objects like normalizers
                    continue
                elif isinstance(v, (str, int, float, bool, type(None))):
                    params_dict[k] = str(v) if v is not None else "None"

            # MLflow has a 100-param limit per batch
            param_items = list(params_dict.items())
            for i in range(0, len(param_items), 100):
                batch = dict(param_items[i:i+100])
                mlflow.log_params(batch)
        except Exception as e:
            print(f"Warning: Could not log all params to MLflow: {e}")

    def _save_args_artifact(self, args, save_path: str):
        """Save args as JSON artifact."""
        try:
            args_dict = {}
            for k, v in vars(args).items():
                if hasattr(v, 'to_dict'):
                    args_dict[k] = v.to_dict()
                elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    args_dict[k] = v
                else:
                    args_dict[k] = str(v)

            args_path = os.path.join(save_path, 'args_mlflow.json')
            os.makedirs(save_path, exist_ok=True)
            with open(args_path, 'w') as f:
                json.dump(args_dict, f, indent=2, default=str)

            mlflow.log_artifact(args_path, artifact_path="config")
        except Exception as e:
            print(f"Warning: Could not save args artifact: {e}")

    def log_metrics(self, metrics: dict, step: int):
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names to values
            step: Training step/epoch
        """
        if self.rank != 0:
            return

        try:
            # MLflow doesn't allow '/' in metric names by default in some contexts
            # But it does support nested naming, so we keep the format
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"Warning: Could not log metrics: {e}")

    def log_model_checkpoint(self, model_path: str, epoch: int):
        """
        Log model checkpoint as artifact.

        Args:
            model_path: Path to saved model checkpoint
            epoch: Current epoch
        """
        if self.rank != 0:
            return

        try:
            mlflow.log_artifact(model_path, artifact_path=f"checkpoints/epoch_{epoch}")
        except Exception as e:
            print(f"Warning: Could not log checkpoint artifact: {e}")

    def finish(self):
        """End the MLflow run."""
        if self.rank != 0:
            return

        try:
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: Could not end MLflow run: {e}")


# Alias for drop-in replacement
TensorBoardLogger = MLflowLogger
