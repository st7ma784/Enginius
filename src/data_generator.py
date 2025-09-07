import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class CarCollectionDataGenerator:
    def __init__(self, seed: int = 42):
        """
        Generate synthetic data for car collection wait times.
        """
        np.random.seed(seed)
        random.seed(seed)

        self.locations = [
            "Downtown",
            "Airport",
            "Mall",
            "Suburb_North",
            "Suburb_South",
            "Industrial",
        ]
        self.drivers = [f"Driver_{i:03d}" for i in range(1, 51)]

        # Base wait times by location (in minutes)
        self.location_base_times = {
            "Downtown": 15,
            "Airport": 25,
            "Mall": 12,
            "Suburb_North": 18,
            "Suburb_South": 20,
            "Industrial": 30,
        }

        # Driver efficiency multipliers
        self.driver_efficiency = {
            driver: np.random.normal(1.0, 0.15) for driver in self.drivers
        }

    def generate_sample(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic car collection data samples."""
        data = []

        for _ in range(n_samples):
            # Random selections
            location = np.random.choice(self.locations)
            driver = np.random.choice(self.drivers)

            # Time of day (hour 0-23)
            hour = np.random.randint(0, 24)

            # Checklist length (number of items to check)
            checklist_length = np.random.poisson(8) + 3  # 3-20 items typically

            # Vehicle type
            is_ev = np.random.choice([True, False], p=[0.3, 0.7])
            needs_fuel = (
                np.random.choice([True, False], p=[0.4, 0.6]) if not is_ev else False
            )
            needs_charge = (
                np.random.choice([True, False], p=[0.6, 0.4]) if is_ev else False
            )

            # Driver needs lunch
            driver_needs_lunch = np.random.choice([True, False], p=[0.2, 0.8])

            # Calculate base wait time
            base_time = self.location_base_times[location]

            # Time of day factor (rush hours take longer)
            time_factor = 1.0
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                time_factor = 1.4
            elif 12 <= hour <= 14:  # Lunch time
                time_factor = 1.2
            elif 22 <= hour or hour <= 6:  # Night/early morning
                time_factor = 0.8

            # Checklist factor
            checklist_factor = (
                1 + (checklist_length - 5) * 0.05
            )  # Each item adds ~3 minutes

            # Vehicle factors
            ev_factor = 1.0
            if is_ev and needs_charge:
                ev_factor = 3.5  # Charging takes much longer
            elif not is_ev and needs_fuel:
                ev_factor = 1.3  # Fueling is quick but adds some time

            # Driver factors
            driver_factor = self.driver_efficiency[driver]
            lunch_factor = 1.5 if driver_needs_lunch else 1.0

            # Calculate expected wait time
            expected_time = (
                base_time
                * time_factor
                * checklist_factor
                * ev_factor
                * driver_factor
                * lunch_factor
            )

            # Add random noise (unmodeled factors)
            actual_wait_time = np.random.gamma(
                shape=expected_time**2 / (expected_time * 0.3) ** 2,
                scale=(expected_time * 0.3) ** 2 / expected_time,
            )

            # Ensure minimum wait time
            actual_wait_time = max(actual_wait_time, 5.0)

            data.append(
                {
                    "location": location,
                    "driver": driver,
                    "hour": hour,
                    "checklist_length": checklist_length,
                    "is_ev": is_ev,
                    "needs_fuel": needs_fuel,
                    "needs_charge": needs_charge,
                    "driver_needs_lunch": driver_needs_lunch,
                    "wait_time_minutes": actual_wait_time,
                }
            )

        return pd.DataFrame(data)

    def get_feature_encodings(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Convert categorical features to numerical encodings."""
        encoded_data = df.copy()
        encodings = {}

        # Location encoding
        location_map = {loc: i for i, loc in enumerate(self.locations)}
        encoded_data["location_encoded"] = df["location"].map(location_map)
        encodings["location"] = location_map

        # Driver encoding
        driver_map = {driver: i for i, driver in enumerate(self.drivers)}
        encoded_data["driver_encoded"] = df["driver"].map(driver_map)
        encodings["driver"] = driver_map

        # Feature matrix
        features = [
            "location_encoded",
            "driver_encoded",
            "hour",
            "checklist_length",
            "is_ev",
            "needs_fuel",
            "needs_charge",
            "driver_needs_lunch",
        ]

        X = encoded_data[features].values.astype(np.float32)
        y = encoded_data["wait_time_minutes"].values.astype(np.float32)

        return X, y, encodings


if __name__ == "__main__":
    # Generate sample data
    generator = CarCollectionDataGenerator()
    df = generator.generate_sample(2000)

    print("Generated Data Sample:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"\nWait time statistics:")
    print(df["wait_time_minutes"].describe())

    # Save sample data
    df.to_csv("../data/car_collection_sample.csv", index=False)
    print("\nSample data saved to ../data/car_collection_sample.csv")
