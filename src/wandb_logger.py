from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb


class WandBLogger:
    def __init__(
        self,
        project_name: str = "car-collection-wait-times",
        entity: str = None,
        config: Dict = None,
    ):
        """
        Initialize WandB logging for the car collection wait time prediction project.

        Args:
            project_name: Name of the WandB project
            entity: WandB entity (username or team)
            config: Configuration dictionary to log
        """
        self.project_name = project_name
        self.entity = entity
        self.run = None
        self.config = config or {}

    def start_run(self, run_name: str = None, tags: List[str] = None):
        """Start a new WandB run."""
        if run_name is None:
            run_name = f"mcmc_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            config=self.config,
            tags=tags or ["mcmc", "bayesian", "wait-time-prediction"],
        )

        return self.run

    def log_data_statistics(self, df: pd.DataFrame):
        """Log dataset statistics to WandB."""
        if self.run is None:
            raise ValueError("Must start a run before logging")

        # Basic statistics
        stats = {
            "data/n_samples": len(df),
            "data/n_locations": df["location"].nunique(),
            "data/n_drivers": df["driver"].nunique(),
            "data/wait_time_mean": df["wait_time_minutes"].mean(),
            "data/wait_time_std": df["wait_time_minutes"].std(),
            "data/wait_time_min": df["wait_time_minutes"].min(),
            "data/wait_time_max": df["wait_time_minutes"].max(),
            "data/ev_percentage": df["is_ev"].mean() * 100,
            "data/needs_fuel_percentage": df["needs_fuel"].mean() * 100,
            "data/needs_charge_percentage": df["needs_charge"].mean() * 100,
            "data/driver_lunch_percentage": df["driver_needs_lunch"].mean() * 100,
        }

        wandb.log(stats)

        # Create and log visualizations
        self._create_data_visualizations(df)

    def _create_data_visualizations(self, df: pd.DataFrame):
        """Create and log data visualization plots."""
        # Wait time distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Histogram of wait times
        axes[0, 0].hist(df["wait_time_minutes"], bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Distribution of Wait Times")
        axes[0, 0].set_xlabel("Wait Time (minutes)")
        axes[0, 0].set_ylabel("Frequency")

        # Wait times by location
        df.boxplot(column="wait_time_minutes", by="location", ax=axes[0, 1])
        axes[0, 1].set_title("Wait Times by Location")
        axes[0, 1].set_xlabel("Location")
        axes[0, 1].set_ylabel("Wait Time (minutes)")
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)

        # Wait times by hour
        hourly_stats = df.groupby("hour")["wait_time_minutes"].mean()
        axes[1, 0].plot(hourly_stats.index, hourly_stats.values, marker="o")
        axes[1, 0].set_title("Average Wait Time by Hour of Day")
        axes[1, 0].set_xlabel("Hour")
        axes[1, 0].set_ylabel("Average Wait Time (minutes)")
        axes[1, 0].grid(True, alpha=0.3)

        # EV vs non-EV wait times
        ev_comparison = df.groupby("is_ev")["wait_time_minutes"].mean()
        axes[1, 1].bar(
            ["Non-EV", "EV"], ev_comparison.values, color=["blue", "green"], alpha=0.7
        )
        axes[1, 1].set_title("Average Wait Time: EV vs Non-EV")
        axes[1, 1].set_ylabel("Average Wait Time (minutes)")

        plt.tight_layout()
        wandb.log({"data_exploration": wandb.Image(fig)})
        plt.close()

    def log_model_config(self, model_params: Dict):
        """Log model configuration parameters."""
        config_dict = {f"model/{k}": v for k, v in model_params.items()}
        wandb.config.update(config_dict)

    def log_mcmc_diagnostics(self, mcmc_results: Dict):
        """Log MCMC diagnostics and convergence metrics."""
        if self.run is None:
            raise ValueError("Must start a run before logging")

        diagnostics = mcmc_results.get("diagnostics", {})

        # Log scalar diagnostics
        for param_name, param_diag in diagnostics.items():
            if isinstance(param_diag, dict):
                for metric_name, metric_value in param_diag.items():
                    if isinstance(metric_value, (int, float)):
                        wandb.log({f"mcmc/{param_name}_{metric_name}": metric_value})

        # Log summary statistics
        summary = mcmc_results.get("summary", {})
        for param_name, param_summary in summary.items():
            if isinstance(param_summary, dict):
                for stat_name, stat_value in param_summary.items():
                    if isinstance(stat_value, (int, float)):
                        wandb.log({f"posterior/{param_name}_{stat_name}": stat_value})

    def log_posterior_distributions(self, samples: Dict, param_names: List[str] = None):
        """Create and log posterior distribution plots."""
        if self.run is None:
            raise ValueError("Must start a run before logging")

        if param_names is None:
            param_names = list(samples.keys())

        # Filter to scalar parameters for visualization
        scalar_params = {}
        for name in param_names:
            param_samples = samples[name].cpu().numpy()
            if len(param_samples.shape) == 1:  # Scalar parameter
                scalar_params[name] = param_samples

        if not scalar_params:
            return

        n_params = len(scalar_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (param_name, param_samples) in enumerate(scalar_params.items()):
            if i < len(axes):
                axes[i].hist(param_samples, bins=50, alpha=0.7, density=True)
                axes[i].set_title(f"Posterior: {param_name}")
                axes[i].set_xlabel("Value")
                axes[i].set_ylabel("Density")
                axes[i].grid(True, alpha=0.3)

                # Add vertical lines for mean and credible intervals
                mean_val = np.mean(param_samples)
                q025 = np.percentile(param_samples, 2.5)
                q975 = np.percentile(param_samples, 97.5)

                axes[i].axvline(
                    mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}"
                )
                axes[i].axvline(
                    q025,
                    color="orange",
                    linestyle=":",
                    label=f"95% CI: [{q025:.3f}, {q975:.3f}]",
                )
                axes[i].axvline(q975, color="orange", linestyle=":")
                axes[i].legend()

        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        wandb.log({"posterior_distributions": wandb.Image(fig)})
        plt.close()

    def log_prediction_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        feature_names: List[str] = None,
    ):
        """Log prediction results and performance metrics."""
        if self.run is None:
            raise ValueError("Must start a run before logging")

        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Log metrics
        metrics = {
            "performance/mse": mse,
            "performance/mae": mae,
            "performance/rmse": rmse,
            "performance/mape": mape,
            "performance/r2": r2,
        }
        wandb.log(metrics)

        # Create prediction plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Predicted vs actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        axes[0, 0].set_xlabel("Actual Wait Time")
        axes[0, 0].set_ylabel("Predicted Wait Time")
        axes[0, 0].set_title(f"Predicted vs Actual (R² = {r2:.3f})")
        axes[0, 0].grid(True, alpha=0.3)

        # Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("Predicted Wait Time")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residual Plot")
        axes[0, 1].grid(True, alpha=0.3)

        # Prediction intervals
        sorted_indices = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_indices]
        y_true_sorted = y_true[sorted_indices]
        y_std_sorted = y_std[sorted_indices]

        axes[1, 0].fill_between(
            y_pred_sorted,
            y_pred_sorted - 1.96 * y_std_sorted,
            y_pred_sorted + 1.96 * y_std_sorted,
            alpha=0.3,
            label="95% Prediction Interval",
        )
        axes[1, 0].scatter(y_pred_sorted, y_true_sorted, alpha=0.6, s=10)
        axes[1, 0].plot(y_pred_sorted, y_pred_sorted, "r--", lw=2)
        axes[1, 0].set_xlabel("Predicted Wait Time")
        axes[1, 0].set_ylabel("Actual Wait Time")
        axes[1, 0].set_title("Predictions with Uncertainty")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Distribution of prediction uncertainties
        axes[1, 1].hist(y_std, bins=30, alpha=0.7, density=True)
        axes[1, 1].set_xlabel("Prediction Standard Deviation")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Distribution of Prediction Uncertainties")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        wandb.log({"prediction_results": wandb.Image(fig)})
        plt.close()

    def log_hyperparameter_tuning(self, tuning_results: Dict):
        """Log hyperparameter tuning results."""
        if self.run is None:
            raise ValueError("Must start a run before logging")

        best_params = tuning_results["best_params"]
        best_score = tuning_results["best_score"]

        # Log best parameters
        for param_name, param_value in best_params.items():
            wandb.log({f"best_params/{param_name}": param_value})

        wandb.log({"best_validation_mse": best_score})

        # Create hyperparameter comparison plot
        results_df = pd.DataFrame(tuning_results["all_results"])

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # MSE by number of samples
        sample_groups = results_df.groupby("num_samples")["avg_mse"].mean()
        axes[0].bar(sample_groups.index.astype(str), sample_groups.values)
        axes[0].set_title("Validation MSE by Number of MCMC Samples")
        axes[0].set_xlabel("Number of Samples")
        axes[0].set_ylabel("Average MSE")

        # MSE by warmup steps
        warmup_groups = results_df.groupby("warmup_steps")["avg_mse"].mean()
        axes[1].bar(warmup_groups.index.astype(str), warmup_groups.values)
        axes[1].set_title("Validation MSE by Warmup Steps")
        axes[1].set_xlabel("Warmup Steps")
        axes[1].set_ylabel("Average MSE")

        plt.tight_layout()
        wandb.log({"hyperparameter_tuning": wandb.Image(fig)})
        plt.close()

    def finish_run(self):
        """Finish the current WandB run."""
        if self.run is not None:
            wandb.finish()
            self.run = None
