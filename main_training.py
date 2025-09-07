#!/usr/bin/env python3
"""
Main training script for the Car Collection Wait Time Prediction system.
This script demonstrates the complete MCMC modeling pipeline with WandB logging.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data_generator import CarCollectionDataGenerator
from mcmc_model import BayesianWaitTimeModel, ModelTuner
from visualizations import PosteriorVisualizer
from wandb_logger import WandBLogger


def main():
    parser = argparse.ArgumentParser(
        description="Train Car Collection Wait Time Prediction Model"
    )
    parser.add_argument(
        "--n_samples", type=int, default=2000, help="Number of training samples"
    )
    parser.add_argument(
        "--mcmc_samples", type=int, default=1000, help="Number of MCMC samples"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="MCMC warmup steps"
    )
    parser.add_argument(
        "--num_chains",
        type=int,
        default=None,
        help="Number of MCMC chains (auto-capped at 8)",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="car-collection-wait-times",
        help="WandB project name",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test set proportion"
    )
    parser.add_argument(
        "--tune_hyperparams", action="store_true", help="Perform hyperparameter tuning"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)

    print("🚗 Car Collection Wait Time Prediction - Training Pipeline")
    print("=" * 60)

    # Initialize components
    print("\n1. Initializing components...")
    generator = CarCollectionDataGenerator(seed=42)
    visualizer = PosteriorVisualizer()

    # Initialize WandB if requested
    wandb_logger = None
    if args.use_wandb:
        print("   - Initializing WandB logging...")
        config = {
            "n_samples": args.n_samples,
            "mcmc_samples": args.mcmc_samples,
            "warmup_steps": args.warmup_steps,
            "num_chains": args.num_chains,
            "test_size": args.test_size,
            "device": args.device,
        }

        wandb_logger = WandBLogger(project_name=args.wandb_project, config=config)
        run = wandb_logger.start_run(
            run_name=f"mcmc_training_{args.mcmc_samples}samples"
        )
        print(f"   - WandB run started: {run.name}")

    # Generate synthetic data
    print(f"\n2. Generating {args.n_samples} synthetic samples...")
    data = generator.generate_sample(args.n_samples)
    print(f"   - Generated data shape: {data.shape}")
    print(
        f"   - Wait time range: {data['wait_time_minutes'].min():.1f} - {data['wait_time_minutes'].max():.1f} minutes"
    )
    print(f"   - Mean wait time: {data['wait_time_minutes'].mean():.1f} minutes")

    # Save data
    data_path = os.path.join(args.output_dir, "training_data.csv")
    data.to_csv(data_path, index=False)
    print(f"   - Data saved to: {data_path}")

    # Log data statistics to WandB
    if wandb_logger:
        print("   - Logging data statistics to WandB...")
        wandb_logger.log_data_statistics(data)

    # Prepare features
    print("\n3. Preparing features...")
    X, y, encodings = generator.get_feature_encodings(data)
    print(f"   - Feature matrix shape: {X.shape}")
    print(f"   - Target vector shape: {y.shape}")
    print(f"   - Number of locations: {len(generator.locations)}")
    print(f"   - Number of drivers: {len(generator.drivers)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")

    # Initialize model
    print(f"\n4. Initializing Bayesian model (device: {args.device})...")
    n_locations = len(generator.locations)
    n_drivers = len(generator.drivers)
    model = BayesianWaitTimeModel(n_locations, n_drivers, device=args.device)

    # Hyperparameter tuning (optional)
    if args.tune_hyperparams:
        print("\n5. Performing hyperparameter tuning...")
        tuner = ModelTuner(model)

        hyperparams_grid = {
            "num_samples": [500, 1000, 1500],
            "warmup_steps": [250, 500, 750],
            "num_chains": [2, 4, 6],  # Will be auto-capped by model
        }

        tuning_results = tuner.cross_validate(
            X_train, y_train, n_folds=3, hyperparams_grid=hyperparams_grid
        )

        print("   - Best hyperparameters:")
        for param, value in tuning_results["best_params"].items():
            print(f"     {param}: {value}")
        print(f"   - Best validation MSE: {tuning_results['best_score']:.3f}")

        # Update arguments with best parameters
        args.mcmc_samples = tuning_results["best_params"]["num_samples"]
        args.warmup_steps = tuning_results["best_params"]["warmup_steps"]
        args.num_chains = tuning_results["best_params"]["num_chains"]

        if wandb_logger:
            wandb_logger.log_hyperparameter_tuning(tuning_results)

    # Train model
    print(f"\n{'5' if not args.tune_hyperparams else '6'}. Training MCMC model...")
    print(f"   - MCMC samples: {args.mcmc_samples}")
    print(f"   - Warmup steps: {args.warmup_steps}")
    print(f"   - Number of chains: {args.num_chains}")

    model_config = {
        "num_samples": args.mcmc_samples,
        "warmup_steps": args.warmup_steps,
        "num_chains": args.num_chains,
        "n_locations": n_locations,
        "n_drivers": n_drivers,
    }

    if wandb_logger:
        wandb_logger.log_model_config(model_config)

    # Fit the model
    results = model.fit(
        X_train,
        y_train,
        num_samples=args.mcmc_samples,
        warmup_steps=args.warmup_steps,
        num_chains=args.num_chains,
    )

    print("   - Model training completed!")

    # Log MCMC diagnostics
    if wandb_logger:
        print("   - Logging MCMC diagnostics to WandB...")
        wandb_logger.log_mcmc_diagnostics(results)

    # Get posterior summary
    print(
        f"\n{'6' if not args.tune_hyperparams else '7'}. Analyzing posterior distributions..."
    )
    posterior_summary = model.get_posterior_summary()

    print("   - Key parameter estimates:")
    key_params = [
        "location_mean",
        "driver_mean",
        "hour_amplitude",
        "checklist_coef",
        "ev_coef",
        "charge_coef",
        "lunch_coef",
        "noise_std",
    ]

    for param in key_params:
        if param in posterior_summary:
            stats = posterior_summary[param]
            print(f"     {param}: {stats['mean']:.3f} ± {stats['std']:.3f}")

    # Create visualizations
    print(f"\n{'7' if not args.tune_hyperparams else '8'}. Creating visualizations...")

    # Posterior distributions
    fig_posterior = visualizer.plot_posterior_distributions(
        results["samples"],
        save_path=os.path.join(
            args.output_dir, "visualizations", "posterior_distributions.png"
        ),
    )
    print("   - Posterior distributions plot saved")

    # Trace plots
    fig_trace = visualizer.plot_trace_plots(
        results["samples"],
        save_path=os.path.join(args.output_dir, "visualizations", "trace_plots.png"),
    )
    print("   - Trace plots saved")

    # Correlation matrix
    fig_corr = visualizer.plot_correlation_matrix(
        results["samples"],
        save_path=os.path.join(
            args.output_dir, "visualizations", "correlation_matrix.png"
        ),
    )
    print("   - Correlation matrix saved")

    # Log visualizations to WandB
    if wandb_logger:
        print("   - Logging posterior distributions to WandB...")
        wandb_logger.log_posterior_distributions(results["samples"])

    # Make predictions on test set
    print(f"\n{'8' if not args.tune_hyperparams else '9'}. Evaluating on test set...")
    y_pred, y_std = model.predict(X_test, num_samples=500)

    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # R-squared
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print("   - Test Set Performance:")
    print(f"     MSE: {mse:.3f}")
    print(f"     MAE: {mae:.3f}")
    print(f"     RMSE: {rmse:.3f}")
    print(f"     MAPE: {mape:.2f}%")
    print(f"     R²: {r2:.3f}")

    # Prediction visualization
    fig_pred = visualizer.plot_prediction_intervals(
        X_test,
        y_test,
        y_pred,
        y_std,
        save_path=os.path.join(
            args.output_dir, "visualizations", "prediction_analysis.png"
        ),
    )
    print("   - Prediction analysis plot saved")

    # Log prediction results
    if wandb_logger:
        print("   - Logging prediction results to WandB...")
        wandb_logger.log_prediction_results(y_test, y_pred, y_std)

    # Save model and results
    print(
        f"\n{'9' if not args.tune_hyperparams else '10'}. Saving model and results..."
    )

    # Save model state
    model_save_path = os.path.join(args.output_dir, "models", "trained_model.pkl")
    import pickle

    with open(model_save_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "results": results,
                "encodings": encodings,
                "config": model_config,
                "posterior_summary": posterior_summary,
            },
            f,
        )
    print(f"   - Model saved to: {model_save_path}")

    # Save results summary
    results_summary = {
        "test_metrics": {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2},
        "model_config": model_config,
        "data_info": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_locations": n_locations,
            "n_drivers": n_drivers,
        },
    }

    results_path = os.path.join(args.output_dir, "results_summary.json")
    import json

    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"   - Results summary saved to: {results_path}")

    # Finish WandB run
    if wandb_logger:
        print("\n   - Finishing WandB run...")
        wandb_logger.finish_run()

    print(f"\n{'='*60}")
    print("🎉 Training pipeline completed successfully!")
    print(f"📁 All outputs saved to: {args.output_dir}")
    print(f"📊 Final test R²: {r2:.3f}")
    print(f"⏱️  Final test RMSE: {rmse:.1f} minutes")


if __name__ == "__main__":
    main()
