from typing import Dict, List, Optional, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from plotly.subplots import make_subplots


class PosteriorVisualizer:
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the posterior visualization suite.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        plt.style.use("default")  # Reset to default first
        self.figsize = figsize
        sns.set_palette("husl")

    def plot_posterior_distributions(
        self, samples: Dict, param_names: List[str] = None, save_path: str = None
    ) -> plt.Figure:
        """
        Create comprehensive posterior distribution plots.

        Args:
            samples: Dictionary of parameter samples from MCMC
            param_names: List of parameter names to plot (None for all scalar params)
            save_path: Path to save the figure
        """
        # Filter to scalar parameters
        if param_names is None:
            param_names = []
            for name, param_samples in samples.items():
                if isinstance(param_samples, torch.Tensor):
                    param_samples = param_samples.cpu().numpy()
                if len(param_samples.shape) == 1:
                    param_names.append(name)

        scalar_samples = {}
        for name in param_names:
            param_data = samples[name]
            if isinstance(param_data, torch.Tensor):
                param_data = param_data.cpu().numpy()
            if len(param_data.shape) == 1:
                scalar_samples[name] = param_data

        if not scalar_samples:
            raise ValueError("No scalar parameters found for visualization")

        n_params = len(scalar_samples)
        n_cols = min(4, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for i, (param_name, param_samples) in enumerate(scalar_samples.items()):
            if i < len(axes):
                ax = axes[i]

                # Histogram with KDE
                ax.hist(
                    param_samples,
                    bins=50,
                    alpha=0.6,
                    density=True,
                    color="skyblue",
                    edgecolor="black",
                    linewidth=0.5,
                )

                # KDE overlay
                try:
                    sns.kdeplot(param_samples, ax=ax, color="red", linewidth=2)
                except:
                    pass

                # Statistics
                mean_val = np.mean(param_samples)
                median_val = np.median(param_samples)
                std_val = np.std(param_samples)
                q025 = np.percentile(param_samples, 2.5)
                q975 = np.percentile(param_samples, 97.5)

                # Vertical lines for statistics
                ax.axvline(
                    mean_val,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_val:.3f}",
                )
                ax.axvline(
                    median_val,
                    color="green",
                    linestyle=":",
                    linewidth=2,
                    label=f"Median: {median_val:.3f}",
                )
                ax.axvline(
                    q025,
                    color="orange",
                    linestyle="-",
                    alpha=0.7,
                    label=f"95% CI: [{q025:.3f}, {q975:.3f}]",
                )
                ax.axvline(q975, color="orange", linestyle="-", alpha=0.7)

                ax.set_title(f"Posterior: {param_name}", fontsize=12, fontweight="bold")
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Posterior Distributions", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_trace_plots(
        self, samples: Dict, param_names: List[str] = None, save_path: str = None
    ) -> plt.Figure:
        """
        Create trace plots for MCMC diagnostics.
        """
        if param_names is None:
            param_names = []
            for name, param_samples in samples.items():
                if isinstance(param_samples, torch.Tensor):
                    param_samples = param_samples.cpu().numpy()
                if len(param_samples.shape) == 1:
                    param_names.append(name)

        scalar_samples = {}
        for name in param_names:
            param_data = samples[name]
            if isinstance(param_data, torch.Tensor):
                param_data = param_data.cpu().numpy()
            if len(param_data.shape) == 1:
                scalar_samples[name] = param_data

        n_params = len(scalar_samples)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 3 * n_params))

        if n_params == 1:
            axes = [axes]

        for i, (param_name, param_samples) in enumerate(scalar_samples.items()):
            axes[i].plot(param_samples, alpha=0.8, linewidth=1)
            axes[i].set_title(f"Trace Plot: {param_name}")
            axes[i].set_xlabel("Iteration")
            axes[i].set_ylabel("Value")
            axes[i].grid(True, alpha=0.3)

            # Add running mean
            running_mean = np.cumsum(param_samples) / np.arange(
                1, len(param_samples) + 1
            )
            axes[i].plot(running_mean, color="red", linewidth=2, label="Running Mean")
            axes[i].legend()

        plt.suptitle("MCMC Trace Plots", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_correlation_matrix(
        self, samples: Dict, param_names: List[str] = None, save_path: str = None
    ) -> plt.Figure:
        """
        Plot correlation matrix of posterior samples.
        """
        if param_names is None:
            param_names = []
            for name, param_samples in samples.items():
                if isinstance(param_samples, torch.Tensor):
                    param_samples = param_samples.cpu().numpy()
                if len(param_samples.shape) == 1:
                    param_names.append(name)

        # Create DataFrame of samples
        sample_data = {}
        for name in param_names:
            param_data = samples[name]
            if isinstance(param_data, torch.Tensor):
                param_data = param_data.cpu().numpy()
            if len(param_data.shape) == 1:
                sample_data[name] = param_data

        df = pd.DataFrame(sample_data)

        # Compute correlation matrix
        corr_matrix = df.corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title("Posterior Correlation Matrix", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_interactive_posterior_plot(
        self, samples: Dict, param_names: List[str] = None
    ) -> go.Figure:
        """
        Create interactive posterior distribution plots using Plotly.
        """
        if param_names is None:
            param_names = []
            for name, param_samples in samples.items():
                if isinstance(param_samples, torch.Tensor):
                    param_samples = param_samples.cpu().numpy()
                if len(param_samples.shape) == 1:
                    param_names.append(name)

        scalar_samples = {}
        for name in param_names:
            param_data = samples[name]
            if isinstance(param_data, torch.Tensor):
                param_data = param_data.cpu().numpy()
            if len(param_data.shape) == 1:
                scalar_samples[name] = param_data

        n_params = len(scalar_samples)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        subplot_titles = [f"Posterior: {name}" for name in scalar_samples.keys()]

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
        )

        for i, (param_name, param_samples) in enumerate(scalar_samples.items()):
            row = i // n_cols + 1
            col = i % n_cols + 1

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=param_samples,
                    nbinsx=50,
                    opacity=0.7,
                    name=f"{param_name} samples",
                    histnorm="probability density",
                ),
                row=row,
                col=col,
            )

            # Add mean and credible interval lines
            mean_val = np.mean(param_samples)
            q025 = np.percentile(param_samples, 2.5)
            q975 = np.percentile(param_samples, 97.5)

            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.3f}",
                row=row,
                col=col,
            )

        fig.update_layout(
            title_text="Interactive Posterior Distributions",
            showlegend=False,
            height=400 * n_rows,
        )

        return fig

    def plot_prediction_intervals(
        self,
        X_test: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        feature_names: List[str] = None,
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot predictions with uncertainty intervals.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Sort by predicted values for cleaner visualization
        sort_idx = np.argsort(y_pred)
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        y_std_sorted = y_std[sort_idx]

        # 1. Predicted vs Actual with uncertainty
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        axes[0, 0].errorbar(
            y_true, y_pred, yerr=y_std, fmt="none", alpha=0.3, capsize=2
        )
        axes[0, 0].plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        axes[0, 0].set_xlabel("Actual Wait Time (minutes)")
        axes[0, 0].set_ylabel("Predicted Wait Time (minutes)")
        axes[0, 0].set_title("Predictions vs Actual with Uncertainty")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Prediction intervals
        x_range = np.arange(len(y_pred_sorted))
        axes[0, 1].fill_between(
            x_range,
            y_pred_sorted - 1.96 * y_std_sorted,
            y_pred_sorted + 1.96 * y_std_sorted,
            alpha=0.3,
            label="95% Prediction Interval",
        )
        axes[0, 1].plot(x_range, y_pred_sorted, "b-", label="Predicted", linewidth=2)
        axes[0, 1].scatter(
            x_range, y_true_sorted, c="red", s=10, alpha=0.7, label="Actual"
        )
        axes[0, 1].set_xlabel("Test Sample (sorted by prediction)")
        axes[0, 1].set_ylabel("Wait Time (minutes)")
        axes[0, 1].set_title("Prediction Intervals")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Residuals vs predicted
        residuals = y_true - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color="r", linestyle="--")
        axes[1, 0].set_xlabel("Predicted Wait Time (minutes)")
        axes[1, 0].set_ylabel("Residuals (minutes)")
        axes[1, 0].set_title("Residuals vs Predictions")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Uncertainty distribution
        axes[1, 1].hist(y_std, bins=30, alpha=0.7, density=True, color="green")
        axes[1, 1].set_xlabel("Prediction Standard Deviation (minutes)")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Distribution of Prediction Uncertainties")
        axes[1, 1].axvline(
            np.mean(y_std),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(y_std):.2f}",
        )
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Model Prediction Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_feature_importance(
        self, samples: Dict, feature_names: List[str], save_path: str = None
    ) -> plt.Figure:
        """
        Plot feature importance based on posterior parameter magnitudes.
        """
        # Extract coefficient samples
        coef_samples = {}
        for name, param_samples in samples.items():
            if "coef" in name.lower() and len(param_samples.shape) == 1:
                if isinstance(param_samples, torch.Tensor):
                    param_samples = param_samples.cpu().numpy()
                coef_samples[name] = param_samples

        if not coef_samples:
            print("No coefficient parameters found for feature importance")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 1. Coefficient magnitudes
        coef_means = []
        coef_stds = []
        coef_names = []

        for name, samples_array in coef_samples.items():
            coef_means.append(np.mean(np.abs(samples_array)))
            coef_stds.append(np.std(samples_array))
            coef_names.append(name.replace("_coef", "").replace("_", " ").title())

        y_pos = np.arange(len(coef_names))
        axes[0].barh(y_pos, coef_means, xerr=coef_stds, alpha=0.7)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(coef_names)
        axes[0].set_xlabel("Mean Absolute Coefficient Value")
        axes[0].set_title("Feature Importance (Coefficient Magnitudes)")
        axes[0].grid(True, alpha=0.3)

        # 2. Coefficient posterior distributions
        for i, (name, samples_array) in enumerate(coef_samples.items()):
            clean_name = name.replace("_coef", "").replace("_", " ").title()
            axes[1].hist(
                samples_array, bins=30, alpha=0.6, label=clean_name, density=True
            )

        axes[1].set_xlabel("Coefficient Value")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Coefficient Posterior Distributions")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
