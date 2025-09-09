# Car Collection Wait Time Predictor

A Bayesian machine learning system for predicting car collection wait times using MCMC methods with PyTorch and Pyro.

## 🎯 Project Overview

This project implements a comprehensive ML framework for predicting wait times for car collection services. The system uses Bayesian MCMC modeling to handle uncertainty and incorporates factors such as:

- **Location/Dealer**: Different pickup locations have varying base wait times
- **Driver Efficiency**: Individual driver performance variations
- **Time of Day**: Rush hour and off-peak time effects
- **Checklist Length**: Number of inspection items affects duration
- **Vehicle Type**: Electric vehicles may need charging (hours vs. minutes for fuel)
- **Driver Needs**: Whether the driver needs lunch breaks

## 🏗️ Architecture

```
├── src/                          # Core source code
│   ├── data_generator.py        # Synthetic data generation
│   ├── mcmc_model.py           # Bayesian MCMC model
│   ├── wandb_logger.py         # Experiment tracking
│   └── visualizations.py       # Posterior visualization suite
├── dashboard/                   # Interactive Streamlit dashboard
│   └── streamlit_app.py        # Web-based model interface
├── tests/                      # Unit tests
├── main_training.py           # Complete training pipeline
└── docker-compose.yml         # Container orchestration
```

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd windsurf-project

# Start the dashboard
docker-compose up wait-time-predictor

# Access dashboard at http://localhost:8501

# Run training pipeline
docker-compose --profile training up trainer
```

### Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python main_training.py --n_samples 2000 --mcmc_samples 1000 --use_wandb

# Start dashboard
streamlit run dashboard/streamlit_app.py
```

## 🔬 Features

### Bayesian MCMC Modeling
- **Hierarchical Bayesian model** with location and driver effects
- **MCMC sampling** using NUTS algorithm via Pyro
- **Uncertainty quantification** with credible intervals
- **Hyperparameter tuning** with cross-validation

### Interactive Dashboard
- **Real-time predictions** with uncertainty estimates
- **Scenario analysis** for different time periods and conditions
- **Model diagnostics** and posterior visualizations
- **Batch prediction** capabilities

### Experiment Tracking
- **Weights & Biases integration** for experiment logging
- **Posterior distribution visualization**
- **Performance metrics tracking**
- **Model artifact storage**

### Production Ready
- **Docker containerization** with health checks
- **GitHub Actions CI/CD** pipeline
- **Automated testing** and code quality checks
- **Scheduled model retraining**

## 📊 Model Details

### Bayesian Model Structure

The model uses a hierarchical Bayesian approach:

```python
# Location effects (hierarchical)
location_mean ~ Normal(20.0, 10.0)
location_std ~ HalfNormal(5.0)
location_effects ~ Normal(location_mean, location_std)

# Driver effects (hierarchical) 
driver_mean ~ Normal(1.0, 0.2)
driver_std ~ HalfNormal(0.3)
driver_effects ~ Normal(driver_mean, driver_std)

# Time of day (sinusoidal)
hour_amplitude ~ HalfNormal(5.0)
hour_phase ~ Uniform(0, 2π)

# Feature coefficients
checklist_coef ~ Normal(1.0, 0.5)
ev_coef ~ Normal(0.0, 2.0)
charge_coef ~ Normal(15.0, 5.0)  # Charging takes much longer
lunch_coef ~ Normal(8.0, 3.0)

# Prediction
μ = location_effect × driver_effect + time_effect + 
    checklist_coef × checklist_length + 
    charge_coef × needs_charge + ...

wait_time ~ LogNormal(μ, σ)
```

### Key Features
- **Uncertainty quantification**: Provides prediction intervals
- **Hierarchical structure**: Shares information across groups
- **Non-linear time effects**: Sinusoidal patterns for rush hours
- **Robust to outliers**: Log-normal likelihood
- **Scalable**: Can handle thousands of locations/drivers

## 🎮 Usage Examples

### Training a Model

```bash
# Basic training
python main_training.py --n_samples 2000 --mcmc_samples 1000

# With hyperparameter tuning
python main_training.py --tune_hyperparams --use_wandb

# Custom configuration
python main_training.py \
  --n_samples 5000 \
  --mcmc_samples 2000 \
  --warmup_steps 1000 \
  --num_chains 4 \
  --use_wandb
```

### Making Predictions

```python
from src.mcmc_model import BayesianWaitTimeModel
from src.data_generator import CarCollectionDataGenerator

# Auto-detect best device (GPU if available, otherwise CPU)
# Automatically caps CPU usage at 8 cores to preserve web interface capacity
model = BayesianWaitTimeModel(n_locations=6, n_drivers=50, device='auto')

# Or specify device explicitly
# model = BayesianWaitTimeModel(n_locations=6, n_drivers=50, device='cuda')  # Use GPU
# model = BayesianWaitTimeModel(n_locations=6, n_drivers=50, device='cpu')   # Use CPU

# Train with multiple chains (auto-capped at 8 for system stability)
results = model.fit(X_train, y_train, 
                   num_samples=1000, 
                   num_chains=4)  # Will use min(4, available_cores, 8)

# Make prediction
prediction_input = [[0, 5, 14, 8, 1, 0, 1, 0]]  # location, driver, hour, etc.
mean_wait, std_wait = model.predict(prediction_input)

print(f"Expected wait time: {mean_wait[0]:.1f} ± {std_wait[0]:.1f} minutes")
```

### Dashboard Usage

1. **Generate Data**: Use sidebar to create synthetic datasets
2. **Train Model**: Configure MCMC parameters and train
3. **Make Predictions**: Input scenario parameters for predictions
4. **Analyze Results**: View posterior distributions and diagnostics
5. **Scenario Testing**: Compare different conditions and time periods

## 📈 Performance

The model achieves strong performance on synthetic data:
- **R² Score**: ~0.75-0.85 on test data
- **RMSE**: ~10-15 minutes prediction error
- **Coverage**: 95% prediction intervals have ~95% actual coverage
- **Convergence**: R̂ < 1.1 for all parameters

## 🔮 Future Enhancements

### Planned Features
- **Real data integration** from car collection services
- **Route optimization** for return journeys
- **Reinforcement learning** comparison with statistical methods
- **Multi-objective optimization** balancing cost vs. wait time
- **Real-time model updates** with streaming data

### RL Integration (Bonus)
The framework is designed to support comparison with reinforcement learning approaches:
- **Travel booking decisions** under uncertainty
- **Driver assignment optimization**
- **Dynamic route planning** based on real-time conditions

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_mcmc_model.py -v
```

### Code Quality

```bash
# Format code
black src/ main_training.py

# Check imports
isort src/ main_training.py

# Lint
flake8 src/ --max-line-length=127
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure CI passes
5. Submit a pull request

## 📄 License

This project is developed for research and educational purposes any commercial applications are to acknowledge Dr Stephen Mander

## 🤝 Acknowledgments

- **PyTorch & Pyro** for probabilistic programming
- **Streamlit** for rapid dashboard development  
- **Weights & Biases** for experiment tracking
- **Docker** for containerization
