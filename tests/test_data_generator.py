import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_generator import CarCollectionDataGenerator


class TestCarCollectionDataGenerator(unittest.TestCase):
    
    def setUp(self):
        self.generator = CarCollectionDataGenerator(seed=42)
    
    def test_initialization(self):
        """Test that the generator initializes correctly."""
        self.assertEqual(len(self.generator.locations), 6)
        self.assertEqual(len(self.generator.drivers), 50)
        self.assertIn('Downtown', self.generator.locations)
        self.assertIn('Airport', self.generator.locations)
    
    def test_generate_sample(self):
        """Test data generation functionality."""
        n_samples = 100
        data = self.generator.generate_sample(n_samples)
        
        # Check data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), n_samples)
        
        # Check required columns
        required_columns = [
            'location', 'driver', 'hour', 'checklist_length',
            'is_ev', 'needs_fuel', 'needs_charge', 'driver_needs_lunch',
            'wait_time_minutes'
        ]
        for col in required_columns:
            self.assertIn(col, data.columns)
        
        # Check data types and ranges
        self.assertTrue(data['hour'].between(0, 23).all())
        self.assertTrue(data['checklist_length'].min() >= 3)
        self.assertTrue(data['wait_time_minutes'].min() >= 5.0)
        self.assertTrue(data['is_ev'].dtype == bool)
    
    def test_feature_encodings(self):
        """Test feature encoding functionality."""
        data = self.generator.generate_sample(50)
        X, y, encodings = self.generator.get_feature_encodings(data)
        
        # Check output shapes
        self.assertEqual(X.shape[0], 50)
        self.assertEqual(X.shape[1], 8)  # 8 features
        self.assertEqual(y.shape[0], 50)
        
        # Check encodings
        self.assertIn('location', encodings)
        self.assertIn('driver', encodings)
        
        # Check data types
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)
    
    def test_reproducibility(self):
        """Test that the generator produces reproducible results."""
        gen1 = CarCollectionDataGenerator(seed=123)
        gen2 = CarCollectionDataGenerator(seed=123)
        
        data1 = gen1.generate_sample(10)
        data2 = gen2.generate_sample(10)
        
        pd.testing.assert_frame_equal(data1, data2)


if __name__ == '__main__':
    unittest.main()