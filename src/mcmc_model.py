import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import os


class BayesianWaitTimeModel:
    def __init__(self, n_locations: int, n_drivers: int, device: str = 'auto'):
        """
        Bayesian model for predicting car collection wait times using MCMC.
        
        Args:
            n_locations: Number of unique locations
            n_drivers: Number of unique drivers
            device: Device to run computations on ('auto', 'cpu', 'cuda', or specific device)
        """
        self.n_locations = n_locations
        self.n_drivers = n_drivers
        
        # Auto-detect best available device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = 'cpu'
                print("Using CPU")
        else:
            self.device = device
            
        self.samples = None
        
        # Set maximum CPU threads to preserve web interface capacity
        self.max_cpu_threads = min(8, os.cpu_count() or 8)
        self.max_chains = min(8, os.cpu_count() or 8)
        
    def model(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Bayesian model definition for wait time prediction.
        
        Args:
            X: Feature matrix [batch_size, n_features]
            y: Target wait times [batch_size] (None for prediction)
        """
        batch_size = X.shape[0]
        
        # Priors for location effects (hierarchical) - more reasonable scales
        location_mean = pyro.sample("location_mean", dist.Normal(25.0, 15.0))
        location_std = pyro.sample("location_std", dist.HalfNormal(10.0))
        
        with pyro.plate("locations", self.n_locations):
            location_effects = pyro.sample(
                "location_effects", 
                dist.Normal(location_mean, location_std)
            )
        
        # Priors for driver effects (multiplicative, centered at 1)
        driver_mean = pyro.sample("driver_mean", dist.Normal(1.0, 0.3))
        driver_std = pyro.sample("driver_std", dist.HalfNormal(0.5))
        
        with pyro.plate("drivers", self.n_drivers):
            driver_effects = pyro.sample(
                "driver_effects",
                dist.Normal(driver_mean, driver_std)
            )
        
        # Time of day effect (sinusoidal) - more reasonable amplitude
        hour_amplitude = pyro.sample("hour_amplitude", dist.HalfNormal(10.0))
        hour_phase = pyro.sample("hour_phase", dist.Uniform(0, 2 * np.pi))
        
        # Other feature coefficients - better scaled
        checklist_coef = pyro.sample("checklist_coef", dist.Normal(2.0, 1.0))  # ~2 min per item
        ev_coef = pyro.sample("ev_coef", dist.Normal(0.0, 5.0))  # EV base difference
        fuel_coef = pyro.sample("fuel_coef", dist.Normal(5.0, 3.0))  # 5 min for fuel
        charge_coef = pyro.sample("charge_coef", dist.Normal(45.0, 15.0))  # 45 min for charging
        lunch_coef = pyro.sample("lunch_coef", dist.Normal(15.0, 5.0))  # 15 min for lunch
        
        # Noise parameters - realistic scale
        noise_std = pyro.sample("noise_std", dist.HalfNormal(10.0))
        
        # Extract features
        location_idx = X[:, 0].long()
        driver_idx = X[:, 1].long()
        hour = X[:, 2]
        checklist_length = X[:, 3]
        is_ev = X[:, 4]
        needs_fuel = X[:, 5]
        needs_charge = X[:, 6]
        driver_needs_lunch = X[:, 7]
        
        # Calculate predictions
        with pyro.plate("data", batch_size):
            # Base location effect
            base_time = location_effects[location_idx]
            
            # Driver efficiency multiplier (centered at 1.0)
            driver_multiplier = driver_effects[driver_idx]
            
            # Time of day effect (additive)
            time_effect = hour_amplitude * torch.sin(2 * np.pi * hour / 24 + hour_phase)
            
            # Additive effects
            checklist_time = checklist_coef * checklist_length
            ev_effect = ev_coef * is_ev
            fuel_time = fuel_coef * needs_fuel
            charge_time = charge_coef * needs_charge
            lunch_time = lunch_coef * driver_needs_lunch
            
            # Combine effects: base * driver_efficiency + additive_factors
            mu = (base_time * driver_multiplier + 
                  time_effect + 
                  checklist_time + 
                  ev_effect + 
                  fuel_time + 
                  charge_time + 
                  lunch_time)
            
            # Ensure positive predictions
            mu_positive = torch.clamp(mu, min=1.0)
            
            # Use Gamma distribution for positive continuous data (better than log-normal)
            # Gamma(concentration, rate) where mean = concentration/rate
            concentration = (mu_positive ** 2) / (noise_std ** 2)
            rate = mu_positive / (noise_std ** 2)
            
            # Likelihood
            if y is not None:
                pyro.sample("obs", dist.Gamma(concentration, rate), obs=y)
            else:
                pred = pyro.sample("pred", dist.Gamma(concentration, rate))
                return pred
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            num_samples: int = 1000, warmup_steps: int = 500,
            num_chains: int = None, disable_progbar: bool = False) -> Dict:
        """
        Fit the model using MCMC sampling.
        
        Args:
            X: Feature matrix
            y: Target wait times
            num_samples: Number of MCMC samples
            warmup_steps: Number of warmup steps
            num_chains: Number of parallel chains (auto-capped at 8 to preserve web interface)
            disable_progbar: Disable progress bar (useful for Streamlit)
        """
        # Auto-determine number of chains if not specified, capped at max_chains
        if num_chains is None:
            num_chains = min(4, self.max_chains)  # Default to 4 chains, but respect cap
        else:
            num_chains = min(num_chains, self.max_chains)
            
        # Set CPU thread limit to preserve web interface capacity
        original_num_threads = torch.get_num_threads()
        torch.set_num_threads(self.max_cpu_threads)
        
        print(f"Using {num_chains} chains with max {self.max_cpu_threads} CPU threads on {self.device}")
        
        try:
            # Convert to tensors
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            
            # Clear previous samples
            pyro.clear_param_store()
            
            # Setup MCMC with disabled signal handling for Streamlit compatibility
            nuts_kernel = NUTS(self.model)
            mcmc = MCMC(
                nuts_kernel,
                num_samples=num_samples,
                warmup_steps=warmup_steps,
                num_chains=num_chains,
                disable_progbar=disable_progbar,
                mp_context="spawn" if num_chains > 1 else None
            )
            
            # Run MCMC with exception handling for signal issues
            try:
                mcmc.run(X_tensor, y_tensor)
            except ValueError as e:
                if "signal only works in main thread" in str(e):
                    # Fallback to single chain if signal handling fails
                    print("Warning: Multi-threading signal issue detected. Falling back to single chain...")
                    mcmc = MCMC(
                        nuts_kernel,
                        num_samples=num_samples,
                        warmup_steps=warmup_steps,
                        num_chains=1,
                        disable_progbar=disable_progbar
                    )
                    mcmc.run(X_tensor, y_tensor)
                else:
                    raise e
            
            # Store samples
            self.samples = mcmc.get_samples()
            
            return {
                'samples': self.samples,
                'summary': mcmc.summary(),
                'diagnostics': mcmc.diagnostics()
            }
            
        finally:
            # Restore original thread count
            torch.set_num_threads(original_num_threads)
    
    def fit_streamlit_safe(self, X: np.ndarray, y: np.ndarray, 
                          num_samples: int = 500, warmup_steps: int = 250) -> Dict:
        """
        Streamlit-safe version of model fitting using single-threaded MCMC.
        This method avoids signal handling issues in Streamlit.
        
        Args:
            X: Feature matrix
            y: Target wait times
            num_samples: Number of MCMC samples
            warmup_steps: Number of warmup steps
        """
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        # Clear previous samples
        pyro.clear_param_store()
        
        # Force single-threaded execution for Streamlit compatibility
        original_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        
        print(f"Streamlit-safe mode: Using 1 chain with 1 CPU thread on {self.device}")
        
        try:
            # Setup single-chain MCMC to avoid multiprocessing issues
            nuts_kernel = NUTS(self.model, adapt_step_size=True, adapt_mass_matrix=True)
            mcmc = MCMC(
                nuts_kernel,
                num_samples=num_samples,
                warmup_steps=warmup_steps,
                num_chains=1,  # Always use single chain for Streamlit
                disable_progbar=True,  # Disable progress bar to avoid conflicts
                disable_validation=False
            )
            
            # Run MCMC
            mcmc.run(X_tensor, y_tensor)
            
            # Store samples
            self.samples = mcmc.get_samples()
            
            return {
                'samples': self.samples,
                'summary': mcmc.summary(),
                'diagnostics': mcmc.diagnostics()
            }
            
        except Exception as e:
            # If still fails, provide informative error
            raise RuntimeError(f"MCMC training failed in Streamlit environment: {str(e)}")
        
        finally:
            # Restore original thread count
            torch.set_num_threads(original_num_threads)
    
    def predict(self, X_new: np.ndarray, num_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using posterior samples.
        
        Args:
            X_new: New feature matrix for prediction
            num_samples: Number of posterior predictive samples
            
        Returns:
            predictions: Mean predictions
            uncertainties: Standard deviations
        """
        if self.samples is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_tensor = torch.tensor(X_new, dtype=torch.float32).to(self.device)
        
        # Sample from posterior predictive
        predictive = pyro.infer.Predictive(
            self.model, 
            posterior_samples=self.samples,
            num_samples=num_samples
        )
        
        with torch.no_grad():
            predictions_samples = predictive(X_tensor)['pred']
            
        predictions = predictions_samples.mean(dim=0).cpu().numpy()
        uncertainties = predictions_samples.std(dim=0).cpu().numpy()
        
        return predictions, uncertainties
    
    def get_posterior_summary(self) -> Dict:
        """Get summary statistics of posterior distributions."""
        if self.samples is None:
            raise ValueError("Model must be fitted before getting posterior summary")
        
        summary = {}
        for param_name, param_samples in self.samples.items():
            param_np = param_samples.cpu().numpy()
            if len(param_np.shape) == 1:  # Scalar parameters
                summary[param_name] = {
                    'mean': np.mean(param_np),
                    'std': np.std(param_np),
                    'q025': np.percentile(param_np, 2.5),
                    'q975': np.percentile(param_np, 97.5)
                }
            else:  # Vector parameters
                summary[param_name] = {
                    'mean': np.mean(param_np, axis=0),
                    'std': np.std(param_np, axis=0),
                    'shape': param_np.shape
                }
        
        return summary


class ModelTuner:
    def __init__(self, model: BayesianWaitTimeModel):
        """
        Framework for tuning MCMC model hyperparameters.
        """
        self.model = model
        self.tuning_results = []
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      n_folds: int = 5, 
                      hyperparams_grid: Dict = None) -> Dict:
        """
        Perform cross-validation with different hyperparameters.
        """
        from sklearn.model_selection import KFold
        
        if hyperparams_grid is None:
            hyperparams_grid = {
                'num_samples': [500, 1000, 1500],
                'warmup_steps': [250, 500, 750],
                'num_chains': [2, 4]
            }
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        results = []
        
        # Grid search
        for num_samples in hyperparams_grid['num_samples']:
            for warmup_steps in hyperparams_grid['warmup_steps']:
                for num_chains in hyperparams_grid['num_chains']:
                    
                    fold_scores = []
                    
                    for train_idx, val_idx in kf.split(X):
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        
                        # Fit model
                        self.model.fit(
                            X_train, y_train,
                            num_samples=num_samples,
                            warmup_steps=warmup_steps,
                            num_chains=num_chains
                        )
                        
                        # Predict
                        y_pred, y_std = self.model.predict(X_val)
                        
                        # Calculate metrics
                        mse = np.mean((y_val - y_pred) ** 2)
                        mae = np.mean(np.abs(y_val - y_pred))
                        
                        fold_scores.append({'mse': mse, 'mae': mae})
                    
                    # Average across folds
                    avg_mse = np.mean([score['mse'] for score in fold_scores])
                    avg_mae = np.mean([score['mae'] for score in fold_scores])
                    
                    results.append({
                        'num_samples': num_samples,
                        'warmup_steps': warmup_steps,
                        'num_chains': num_chains,
                        'avg_mse': avg_mse,
                        'avg_mae': avg_mae,
                        'fold_scores': fold_scores
                    })
        
        self.tuning_results = results
        best_result = min(results, key=lambda x: x['avg_mse'])
        
        return {
            'best_params': {
                k: best_result[k] for k in ['num_samples', 'warmup_steps', 'num_chains']
            },
            'best_score': best_result['avg_mse'],
            'all_results': results
        }