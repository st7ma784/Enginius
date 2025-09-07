#!/usr/bin/env python3
"""
Quick test script to validate the improved MCMC model architecture.
This tests the model with a small sample size and evaluates predictions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_generator import CarCollectionDataGenerator
from mcmc_model import BayesianWaitTimeModel

def main():
    print("🚗 Testing Improved MCMC Model Architecture")
    print("=" * 50)
    
    # Generate small test dataset
    print("\n1. Generating test data...")
    generator = CarCollectionDataGenerator(seed=42)
    data = generator.generate_sample(200)  # Small sample for quick test
    
    print(f"   - Generated {len(data)} samples")
    print(f"   - Wait time range: {data['wait_time_minutes'].min():.1f} - {data['wait_time_minutes'].max():.1f} minutes")
    print(f"   - Mean wait time: {data['wait_time_minutes'].mean():.1f} minutes")
    
    # Prepare features
    print("\n2. Preparing features...")
    X, y, encodings = generator.get_feature_encodings(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    
    # Initialize and train model
    print("\n3. Training improved model...")
    n_locations = len(generator.locations)
    n_drivers = len(generator.drivers)
    model = BayesianWaitTimeModel(n_locations, n_drivers, device='auto')
    
    # Use smaller sample size for quick test
    results = model.fit(
        X_train, y_train,
        num_samples=100,
        warmup_steps=50,
        num_chains=1,
        disable_progbar=True
    )
    
    print("   - Training completed!")
    
    # Make predictions
    print("\n4. Making predictions...")
    y_pred, y_std = model.predict(X_test, num_samples=100)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Check if predictions are in reasonable range
    pred_ratio = np.mean(y_pred) / np.mean(y_test)
    
    print(f"   - MSE: {mse:.2f}")
    print(f"   - MAE: {mae:.2f}")
    print(f"   - Mean prediction: {np.mean(y_pred):.1f} minutes")
    print(f"   - Mean actual: {np.mean(y_test):.1f} minutes")
    print(f"   - Prediction ratio (pred/actual): {pred_ratio:.3f}")
    
    # Evaluate model architecture improvements
    print("\n5. Model Architecture Assessment:")
    
    if pred_ratio < 0.5:
        print("   ⚠️  WARNING: Predictions are much smaller than actual values")
        print("       This suggests the model architecture still needs improvement")
    elif pred_ratio > 2.0:
        print("   ⚠️  WARNING: Predictions are much larger than actual values")
        print("       This suggests the model architecture needs adjustment")
    else:
        print("   ✅ Predictions are in reasonable range!")
        print("       Model architecture improvements appear successful")
    
    if mae < 20:
        print("   ✅ Mean Absolute Error is reasonable (< 20 minutes)")
    else:
        print("   ⚠️  Mean Absolute Error is high (> 20 minutes)")
        print("       Further model tuning may be needed")
    
    # Check posterior parameter estimates
    print("\n6. Key Parameter Estimates:")
    posterior = model.get_posterior_summary()
    
    key_params = {
        'location_mean': 'Base location wait time',
        'charge_coef': 'EV charging time coefficient',
        'lunch_coef': 'Driver lunch time coefficient',
        'noise_std': 'Model uncertainty'
    }
    
    for param, description in key_params.items():
        if param in posterior:
            stats = posterior[param]
            print(f"   - {description}: {stats['mean']:.1f} ± {stats['std']:.1f} minutes")
    
    # Create simple visualization
    print("\n7. Creating visualization...")
    
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Wait Time (minutes)')
    plt.ylabel('Predicted Wait Time (minutes)')
    plt.title('Actual vs Predicted Wait Times')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Wait Time (minutes)')
    plt.ylabel('Residuals (minutes)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/model_validation.png', dpi=150, bbox_inches='tight')
    print("   - Visualization saved to: outputs/model_validation.png")
    
    print("\n8. Summary:")
    print("   - Model training completed successfully")
    print("   - Architecture improvements implemented:")
    print("     * Better-scaled priors (location_mean=25±15, charge_coef=45±15)")
    print("     * Gamma distribution instead of log-normal for positive data")
    print("     * Realistic coefficient values for charging and lunch times")
    
    if pred_ratio > 0.7 and pred_ratio < 1.5 and mae < 25:
        print("   ✅ OVERALL: Model architecture improvements successful!")
        print("       Ready for dashboard testing")
    else:
        print("   ⚠️  OVERALL: Model may need further refinement")
        print("       Consider additional architecture adjustments")
    
    return model, results, {
        'mse': mse,
        'mae': mae,
        'pred_ratio': pred_ratio,
        'mean_pred': np.mean(y_pred),
        'mean_actual': np.mean(y_test)
    }

if __name__ == "__main__":
    model, results, metrics = main()