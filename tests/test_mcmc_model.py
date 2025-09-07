import unittest
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from mcmc_model import BayesianWaitTimeModel
from data_generator import CarCollectionDataGenerator


class TestBayesianWaitTimeModel(unittest.TestCase):
    
    def setUp(self):
        self.n_locations = 3
        self.n_drivers = 5
        # Force CPU usage for tests to ensure consistency
        self.model = BayesianWaitTimeModel(self.n_locations, self.n_drivers, device='cpu')
        
        # Generate small test dataset
        self.generator = CarCollectionDataGenerator(seed=42)
        data = self.generator.generate_sample(50)
        self.X, self.y, self.encodings = self.generator.get_feature_encodings(data)
        
        # Adjust for smaller test size
        self.X[:, 0] = self.X[:, 0] % self.n_locations  # Limit locations
        self.X[:, 1] = self.X[:, 1] % self.n_drivers    # Limit drivers
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertEqual(self.model.n_locations, self.n_locations)
        self.assertEqual(self.model.n_drivers, self.n_drivers)
        self.assertEqual(self.model.device, 'cpu')
        self.assertIsNone(self.model.samples)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        X_tensor = torch.tensor(self.X[:5], dtype=torch.float32)
        
        # Test model call (should not raise exception)
        try:
            result = self.model.model(X_tensor)
            # If no exception, the forward pass works
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Model forward pass failed: {e}")
    
    def test_fit_basic(self):
        """Test basic model fitting."""
        # Use very small parameters for quick test
        results = self.model.fit(
            self.X, self.y, 
            num_samples=10, 
            warmup_steps=5, 
            num_chains=1
        )
        
        # Check that samples were generated
        self.assertIsNotNone(self.model.samples)
        self.assertIn('samples', results)
        self.assertIn('summary', results)
        
        # Check sample structure
        samples = results['samples']
        expected_params = ['location_mean', 'driver_mean', 'hour_amplitude', 'noise_std']
        for param in expected_params:
            if param in samples:
                self.assertEqual(len(samples[param]), 10)  # num_samples
    
    def test_predict_without_fitting(self):
        """Test that prediction fails without fitting."""
        with self.assertRaises(ValueError):
            self.model.predict(self.X[:5])
    
    def test_predict_after_fitting(self):
        """Test prediction after fitting."""
        # Fit model
        self.model.fit(
            self.X, self.y, 
            num_samples=10, 
            warmup_steps=5, 
            num_chains=1
        )
        
        # Make predictions
        X_test = self.X[:5]
        pred_mean, pred_std = self.model.predict(X_test, num_samples=5)
        
        # Check predictions
        self.assertEqual(len(pred_mean), 5)
        self.assertEqual(len(pred_std), 5)
        self.assertTrue(all(pred_mean > 0))  # Wait times should be positive
        self.assertTrue(all(pred_std > 0))   # Uncertainty should be positive
    
    def test_posterior_summary(self):
        """Test posterior summary generation."""
        # Fit model first
        self.model.fit(
            self.X, self.y, 
            num_samples=10, 
            warmup_steps=5, 
            num_chains=1
        )
        
        summary = self.model.get_posterior_summary()
        
        # Check that summary contains expected statistics
        for param_name, param_stats in summary.items():
            if isinstance(param_stats, dict) and 'mean' in param_stats:
                self.assertIn('mean', param_stats)
                self.assertIn('std', param_stats)
                self.assertIn('q025', param_stats)
                self.assertIn('q975', param_stats)


if __name__ == '__main__':
    unittest.main()