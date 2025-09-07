import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import CarCollectionDataGenerator
from mcmc_model import BayesianWaitTimeModel, ModelTuner
from visualizations import PosteriorVisualizer
from wandb_logger import WandBLogger
import torch
import pickle
from datetime import datetime, timedelta


class WaitTimeDashboard:
    def __init__(self):
        """Initialize the wait time prediction dashboard."""
        self.generator = CarCollectionDataGenerator(seed=42)
        self.model = None
        self.model_trained = False
        self.sample_data = None
        
    def generate_data(self, n_samples: int = 1000):
        """Generate synthetic data for the dashboard."""
        self.sample_data = self.generator.generate_sample(n_samples)
        return self.sample_data
    
    def train_model(self, data: pd.DataFrame, progress_bar=None):
        """Train the MCMC model."""
        X, y, encodings = self.generator.get_feature_encodings(data)
        
        # Initialize model with auto device detection
        n_locations = len(self.generator.locations)
        n_drivers = len(self.generator.drivers)
        self.model = BayesianWaitTimeModel(n_locations, n_drivers, device='auto')
        
        # Train model with Streamlit-compatible settings
        if progress_bar:
            progress_bar.progress(0.3, "Training MCMC model...")
        
        results = self.model.fit_streamlit_safe(
            X, y, 
            num_samples=500, 
            warmup_steps=250
        )
        
        if progress_bar:
            progress_bar.progress(1.0, "Model training complete!")
        
        self.model_trained = True
        return results, encodings
    
    def make_predictions(self, X_new: np.ndarray):
        """Make predictions using the trained model."""
        if not self.model_trained or self.model is None or self.model.samples is None:
            return None, None
        
        try:
            return self.model.predict(X_new, num_samples=200)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None
    
    def handle_unknown_values(self, location=None, driver=None, hour=None, 
                            checklist_length=None, is_ev=None, needs_fuel=None, 
                            needs_charge=None, driver_needs_lunch=None):
        """Handle unknown values by using reasonable defaults or sampling from priors."""
        # Handle unknown location
        if location == "Unknown" or location is None:
            location_idx = np.random.choice(len(self.generator.locations))
        else:
            location_idx = self.generator.locations.index(location)
        
        # Handle unknown driver
        if driver == "Unknown" or driver is None:
            driver_idx = np.random.choice(len(self.generator.drivers))
        else:
            driver_idx = self.generator.drivers.index(driver)
        
        # Handle unknown hour
        if hour is None:
            hour = np.random.randint(6, 22)  # Reasonable business hours
        
        # Handle unknown checklist length
        if checklist_length is None:
            checklist_length = np.random.poisson(8) + 3  # Based on data generator
        
        # Handle unknown vehicle flags
        if is_ev is None:
            is_ev = np.random.choice([True, False], p=[0.3, 0.7])
        
        if needs_fuel is None:
            needs_fuel = np.random.choice([True, False], p=[0.4, 0.6]) if not is_ev else False
        
        if needs_charge is None:
            needs_charge = np.random.choice([True, False], p=[0.6, 0.4]) if is_ev else False
        
        if driver_needs_lunch is None:
            driver_needs_lunch = np.random.choice([True, False], p=[0.2, 0.8])
        
        return np.array([[
            location_idx, driver_idx, hour, checklist_length,
            int(is_ev), int(needs_fuel), int(needs_charge), int(driver_needs_lunch)
        ]], dtype=np.float32)


def main():
    st.set_page_config(
        page_title="Car Collection Wait Time Predictor",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = WaitTimeDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Title and description
    st.title("🚗 Car Collection Wait Time Predictor")
    st.markdown("""
    This dashboard demonstrates a Bayesian MCMC model for predicting car collection wait times.
    The model considers factors like location, driver, time of day, vehicle type, and checklist length.
    """)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Data generation section
    st.sidebar.subheader("Data Generation")
    n_samples = st.sidebar.slider("Number of samples", 500, 5000, 2000, 500)
    
    if st.sidebar.button("Generate New Data"):
        with st.spinner("Generating synthetic data..."):
            dashboard.generate_data(n_samples)
        st.success(f"Generated {n_samples} samples!")
    
    # Generate initial data if none exists
    if dashboard.sample_data is None:
        dashboard.generate_data(n_samples)
    
    # Model training section
    st.sidebar.subheader("Model Training")
    
    if st.sidebar.button("Train MCMC Model", disabled=dashboard.sample_data is None):
        st.info("ℹ️ **Quick Training**: Uses single-chain MCMC for fast, stable training. For advanced multi-chain training with GPU acceleration, see 'Advanced Training Options' below.")
        with st.spinner("Training Bayesian model... This may take a few minutes."):
            progress_bar = st.progress(0, "Starting model training...")
            try:
                results, encodings = dashboard.train_model(dashboard.sample_data, progress_bar)
                st.session_state.model_results = results
                st.session_state.encodings = encodings
                st.success("Model training completed!")
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.info("💡 Tip: Try reducing the number of samples or use the command-line training script.")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Explorer", 
        "🔬 Model Results", 
        "🔮 Predictions", 
        "📈 Scenario Analysis",
        "⚙️ Model Tuning"
    ])
    
    with tab1:
        st.header("Data Exploration")
        
        if dashboard.sample_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Overview")
                st.dataframe(dashboard.sample_data.head(10))
                
                st.subheader("Summary Statistics")
                st.dataframe(dashboard.sample_data.describe())
            
            with col2:
                st.subheader("Wait Time Distribution")
                fig_hist = px.histogram(
                    dashboard.sample_data, 
                    x='wait_time_minutes',
                    nbins=50,
                    title="Distribution of Wait Times"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.subheader("Wait Time by Location")
                fig_box = px.box(
                    dashboard.sample_data,
                    x='location',
                    y='wait_time_minutes',
                    title="Wait Times by Location"
                )
                fig_box.update_xaxes(tickangle=45)
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Time series analysis
            st.subheader("Wait Time Patterns")
            col3, col4 = st.columns(2)
            
            with col3:
                hourly_avg = dashboard.sample_data.groupby('hour')['wait_time_minutes'].mean().reset_index()
                fig_hourly = px.line(
                    hourly_avg,
                    x='hour',
                    y='wait_time_minutes',
                    title="Average Wait Time by Hour of Day",
                    markers=True
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with col4:
                ev_comparison = dashboard.sample_data.groupby('is_ev')['wait_time_minutes'].mean().reset_index()
                ev_comparison['Vehicle Type'] = ev_comparison['is_ev'].map({True: 'Electric', False: 'Conventional'})
                fig_ev = px.bar(
                    ev_comparison,
                    x='Vehicle Type',
                    y='wait_time_minutes',
                    title="Average Wait Time: EV vs Conventional",
                    color='Vehicle Type'
                )
                st.plotly_chart(fig_ev, use_container_width=True)
        
        else:
            st.info("Generate data using the sidebar controls to explore the dataset.")
    
    with tab2:
        st.header("Model Results & Diagnostics")
        
        if hasattr(st.session_state, 'model_results') and dashboard.model_trained:
            results = st.session_state.model_results
            
            # Data vs Model Comparison Section
            st.subheader("📊 Data vs Model Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Actual Data Distribution**")
                if dashboard.sample_data is not None:
                    fig_data = px.histogram(
                        dashboard.sample_data, 
                        x='wait_time_minutes',
                        nbins=30,
                        title="Observed Wait Times",
                        color_discrete_sequence=['skyblue']
                    )
                    fig_data.update_layout(height=300)
                    st.plotly_chart(fig_data, use_container_width=True)
            
            with col2:
                st.write("**Model Predicted Distribution**")
                if dashboard.sample_data is not None and dashboard.model_trained:
                    try:
                        X, y, _ = dashboard.generator.get_feature_encodings(dashboard.sample_data)
                        y_pred, y_std = dashboard.make_predictions(X)
                        
                        if y_pred is not None:
                            fig_pred = px.histogram(
                                x=y_pred,
                                nbins=30,
                                title="Model Predictions",
                                color_discrete_sequence=['orange']
                            )
                            fig_pred.update_layout(height=300)
                            st.plotly_chart(fig_pred, use_container_width=True)
                        else:
                            st.error("Failed to generate predictions")
                    except Exception as e:
                        st.error(f"Error generating predictions: {e}")
            
            # 95% Confidence Interval Explanation
            st.subheader("📈 95% Confidence Interval Explanation")
            
            if dashboard.sample_data is not None and dashboard.model_trained:
                try:
                    # Generate sample predictions for visualization
                    sample_size = min(100, len(dashboard.sample_data))
                    X_sample = X[:sample_size]
                    y_true_sample = y[:sample_size]
                    y_pred_sample, y_std_sample = dashboard.make_predictions(X_sample)
                    
                    if y_pred_sample is not None:
                        # Sort by prediction for cleaner visualization
                        sort_idx = np.argsort(y_pred_sample)
                        y_pred_sorted = y_pred_sample[sort_idx]
                        y_true_sorted = y_true_sample[sort_idx]
                        y_std_sorted = y_std_sample[sort_idx]
                        
                        # Calculate confidence intervals
                        ci_lower = y_pred_sorted - 1.96 * y_std_sorted
                        ci_upper = y_pred_sorted + 1.96 * y_std_sorted
                        
                        # Create confidence interval plot
                        fig_ci = go.Figure()
                        
                        # Add confidence interval band
                        fig_ci.add_trace(go.Scatter(
                            x=list(range(len(y_pred_sorted))) + list(range(len(y_pred_sorted)))[::-1],
                            y=list(ci_upper) + list(ci_lower)[::-1],
                            fill='toself',
                            fillcolor='rgba(135, 206, 235, 0.3)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='95% Confidence Interval',
                            showlegend=True
                        ))
                        
                        # Add predictions
                        fig_ci.add_trace(go.Scatter(
                            x=list(range(len(y_pred_sorted))),
                            y=y_pred_sorted,
                            mode='lines',
                            name='Predicted Wait Time',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Add actual values
                        fig_ci.add_trace(go.Scatter(
                            x=list(range(len(y_true_sorted))),
                            y=y_true_sorted,
                            mode='markers',
                            name='Actual Wait Time',
                            marker=dict(color='red', size=4)
                        ))
                        
                        fig_ci.update_layout(
                            title="95% Confidence Intervals for Wait Time Predictions",
                            xaxis_title="Sample Index (sorted by prediction)",
                            yaxis_title="Wait Time (minutes)",
                            height=400
                        )
                        
                        st.plotly_chart(fig_ci, use_container_width=True)
                        
                        # Calculate coverage
                        coverage = np.mean((y_true_sorted >= ci_lower) & (y_true_sorted <= ci_upper))
                        st.info(f"📊 **95% CI Coverage**: {coverage:.1%} of actual values fall within the confidence interval")
                        
                        st.markdown("""
                        **What does the 95% Confidence Interval mean?**
                        - The shaded blue area represents the range where we expect 95% of predictions to fall
                        - Red dots outside the band indicate cases where the model was less certain
                        - A good model should have ~95% of actual values (red dots) within the blue band
                        """)
                    
                except Exception as e:
                    st.error(f"Error creating CI visualization: {e}")
            
            # Convergence Diagnostics
            st.subheader("🔍 Convergence Diagnostics")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.write("**Parameter Summary**")
                summary = dashboard.model.get_posterior_summary()
                
                # Display key parameters
                key_params = ['location_mean', 'driver_mean', 'hour_amplitude', 
                             'checklist_coef', 'ev_coef', 'charge_coef', 'lunch_coef', 'noise_std']
                
                summary_data = []
                for param in key_params:
                    if param in summary:
                        summary_data.append({
                            'Parameter': param.replace('_', ' ').title(),
                            'Mean': f"{summary[param]['mean']:.3f}",
                            'Std': f"{summary[param]['std']:.3f}",
                            '95% CI': f"[{summary[param]['q025']:.3f}, {summary[param]['q975']:.3f}]"
                        })
                
                if summary_data:
                    st.dataframe(pd.DataFrame(summary_data))
            
            with col4:
                st.write("**Convergence Metrics**")
                diagnostics = results['diagnostics']
                
                convergence_data = []
                n_eff_total = 0
                r_hat_issues = 0
                
                for param_name, diag_dict in diagnostics.items():
                    if isinstance(diag_dict, dict):
                        r_hat = diag_dict.get('r_hat', None)
                        n_eff = diag_dict.get('n_eff', None)
                        
                        if r_hat is not None and isinstance(r_hat, (int, float)):
                            status = "✅" if r_hat < 1.1 else "⚠️"
                            if r_hat >= 1.1:
                                r_hat_issues += 1
                            
                            convergence_data.append({
                                'Parameter': param_name.replace('_', ' ').title()[:15],
                                'R̂': f"{r_hat:.3f}",
                                'Status': status
                            })
                        
                        if n_eff is not None and isinstance(n_eff, (int, float)):
                            n_eff_total += max(0, n_eff)
                
                if convergence_data:
                    st.dataframe(pd.DataFrame(convergence_data[:10]))  # Show first 10
                    
                    # Overall diagnostics
                    st.write("**Overall Diagnostics:**")
                    if r_hat_issues == 0:
                        st.success("✅ All parameters converged (R̂ < 1.1)")
                    else:
                        st.warning(f"⚠️ {r_hat_issues} parameters may not have converged")
                    
                    if n_eff_total > 0:
                        st.info(f"📊 Average effective sample size: {n_eff_total/len(convergence_data):.0f}")
            
            # Bayesian Model Uncertainties
            st.subheader("🎲 Bayesian Model Uncertainties")
            
            st.markdown("""
            **Sources of Uncertainty in the Model:**
            
            1. **Aleatoric Uncertainty** (Data noise):
               - Random variations in wait times even with identical conditions
               - Captured by the `noise_std` parameter
            
            2. **Epistemic Uncertainty** (Model uncertainty):
               - Uncertainty about parameter values (location effects, driver effects)
               - Shown in the width of posterior distributions
            
            3. **Unknown Factors**:
               - Traffic conditions, weather, vehicle maintenance issues
               - Customer-specific needs, unforeseen delays
               - Modeled through hierarchical priors and random effects
            """)
            
            # Show uncertainty quantification
            if 'noise_std' in summary:
                noise_uncertainty = summary['noise_std']['mean']
                st.metric("Data Noise (Aleatoric)", f"{noise_uncertainty:.2f} minutes", 
                         help="Average unpredictable variation in wait times")
            
            # Parameter uncertainty visualization
            st.subheader("📊 Posterior Distributions")
            
            samples = results['samples']
            scalar_params = []
            for name, param_samples in samples.items():
                if isinstance(param_samples, torch.Tensor):
                    param_samples = param_samples.cpu().numpy()
                if len(param_samples.shape) == 1:
                    scalar_params.append(name)
            
            if scalar_params:
                selected_params = st.multiselect(
                    "Select parameters to visualize:",
                    scalar_params,
                    default=[p for p in scalar_params if any(key in p for key in ['location_mean', 'driver_mean', 'charge_coef', 'lunch_coef'])][:4]
                )
                
                if selected_params:
                    n_params = len(selected_params)
                    n_cols = min(2, n_params)
                    n_rows = (n_params + n_cols - 1) // n_cols
                    
                    fig = make_subplots(
                        rows=n_rows, cols=n_cols,
                        subplot_titles=[f'Posterior: {name.replace("_", " ").title()}' for name in selected_params]
                    )
                    
                    for i, param_name in enumerate(selected_params):
                        row = i // n_cols + 1
                        col = i % n_cols + 1
                        
                        param_data = samples[param_name]
                        if isinstance(param_data, torch.Tensor):
                            param_data = param_data.cpu().numpy()
                        
                        fig.add_trace(
                            go.Histogram(
                                x=param_data,
                                nbinsx=30,
                                name=param_name,
                                opacity=0.7,
                                histnorm='probability density'
                            ),
                            row=row, col=col
                        )
                    
                    fig.update_layout(
                        height=400 * n_rows,
                        title_text="Parameter Posterior Distributions (Uncertainty)",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("💡 **Wider distributions = more uncertainty**. Narrow peaks indicate the model is confident about parameter values.")
        
        else:
            st.info("Train the model using the sidebar controls to view results.")
    
    with tab3:
        st.header("Make Predictions")
        
        if dashboard.model_trained:
            st.subheader("Single Prediction")
            
            st.info("💡 Select 'Unknown' for any factor you're uncertain about - the model will use reasonable defaults based on typical patterns.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Input controls with Unknown options
                location_options = ["Unknown"] + dashboard.generator.locations
                selected_location = st.selectbox("Location", location_options)
                
                driver_options = ["Unknown"] + dashboard.generator.drivers[:10]  # Limit for UI
                selected_driver = st.selectbox("Driver", driver_options)
                
                # Hour with Unknown option
                hour_unknown = st.checkbox("Unknown Time of Day")
                if not hour_unknown:
                    selected_hour = st.slider("Hour of Day", 0, 23, 12)
                else:
                    selected_hour = None
                    st.info("Will use typical business hours (6 AM - 10 PM)")
                
                # Checklist length with Unknown option
                checklist_unknown = st.checkbox("Unknown Checklist Length")
                if not checklist_unknown:
                    checklist_length = st.slider("Checklist Length", 3, 20, 8)
                else:
                    checklist_length = None
                    st.info("Will use typical checklist length (~8 items)")
                
            with col2:
                # Vehicle type with Unknown options
                vehicle_type = st.selectbox("Vehicle Type", ["Unknown", "Electric", "Conventional"])
                
                if vehicle_type == "Electric":
                    is_ev = True
                    needs_fuel = False
                    charge_unknown = st.checkbox("Unknown Charging Need")
                    if not charge_unknown:
                        needs_charge = st.checkbox("Needs Charging")
                    else:
                        needs_charge = None
                elif vehicle_type == "Conventional":
                    is_ev = False
                    needs_charge = False
                    fuel_unknown = st.checkbox("Unknown Fuel Need")
                    if not fuel_unknown:
                        needs_fuel = st.checkbox("Needs Fuel")
                    else:
                        needs_fuel = None
                else:  # Unknown
                    is_ev = None
                    needs_fuel = None
                    needs_charge = None
                
                # Driver lunch with Unknown option
                lunch_unknown = st.checkbox("Unknown Lunch Need")
                if not lunch_unknown:
                    driver_needs_lunch = st.checkbox("Driver Needs Lunch")
                else:
                    driver_needs_lunch = None
            
            if st.button("Make Prediction"):
                try:
                    # Handle unknown values
                    X_pred = dashboard.handle_unknown_values(
                        location=selected_location if selected_location != "Unknown" else None,
                        driver=selected_driver if selected_driver != "Unknown" else None,
                        hour=selected_hour,
                        checklist_length=checklist_length,
                        is_ev=is_ev,
                        needs_fuel=needs_fuel,
                        needs_charge=needs_charge,
                        driver_needs_lunch=driver_needs_lunch
                    )
                    
                    # Make prediction
                    pred_mean, pred_std = dashboard.make_predictions(X_pred)
                    
                    if pred_mean is not None and pred_std is not None:
                        # Multiple predictions for unknown values (Monte Carlo)
                        if any(x is None for x in [selected_location == "Unknown", selected_driver == "Unknown", 
                                                  selected_hour is None, checklist_length is None,
                                                  is_ev is None, needs_fuel is None, needs_charge is None,
                                                  driver_needs_lunch is None]):
                            st.info("🎲 **Note**: Some values are unknown, running multiple scenarios...")
                            
                            # Generate multiple predictions with different unknown values
                            predictions = []
                            for i in range(10):
                                X_sample = dashboard.handle_unknown_values(
                                    location=selected_location if selected_location != "Unknown" else None,
                                    driver=selected_driver if selected_driver != "Unknown" else None,
                                    hour=selected_hour,
                                    checklist_length=checklist_length,
                                    is_ev=is_ev,
                                    needs_fuel=needs_fuel,
                                    needs_charge=needs_charge,
                                    driver_needs_lunch=driver_needs_lunch
                                )
                                pred_i, _ = dashboard.make_predictions(X_sample)
                                if pred_i is not None:
                                    predictions.append(pred_i[0])
                            
                            if predictions:
                                mean_pred = np.mean(predictions)
                                std_pred = np.std(predictions)
                                
                                col_pred1, col_pred2 = st.columns(2)
                                with col_pred1:
                                    st.metric("Average Prediction", f"{mean_pred:.1f} minutes")
                                with col_pred2:
                                    st.metric("Uncertainty Range", f"± {std_pred:.1f} minutes")
                                
                                # Show distribution of predictions
                                fig = px.histogram(
                                    x=predictions,
                                    nbins=8,
                                    title="Distribution of Predictions with Unknown Values",
                                    labels={'x': 'Wait Time (minutes)', 'y': 'Frequency'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            # Single prediction with known values
                            st.success(f"**Predicted Wait Time:** {pred_mean[0]:.1f} ± {pred_std[0]:.1f} minutes")
                            
                            # Confidence interval
                            ci_lower = pred_mean[0] - 1.96 * pred_std[0]
                            ci_upper = pred_mean[0] + 1.96 * pred_std[0]
                            st.info(f"**95% Confidence Interval:** [{max(0, ci_lower):.1f}, {ci_upper:.1f}] minutes")
                            
                            # Show input summary
                            with st.expander("Input Summary"):
                                st.write(f"- Location: {selected_location}")
                                st.write(f"- Driver: {selected_driver}")
                                st.write(f"- Time: {selected_hour}:00" if selected_hour is not None else "Unknown")
                                st.write(f"- Checklist Length: {checklist_length}" if checklist_length is not None else "Unknown")
                                st.write(f"- Vehicle Type: {vehicle_type}")
                                if vehicle_type != "Unknown":
                                    if is_ev:
                                        st.write(f"- Needs Charging: {'Yes' if needs_charge else 'No' if needs_charge is not None else 'Unknown'}")
                                    else:
                                        st.write(f"- Needs Fuel: {'Yes' if needs_fuel else 'No' if needs_fuel is not None else 'Unknown'}")
                                st.write(f"- Driver Lunch: {'Yes' if driver_needs_lunch else 'No' if driver_needs_lunch is not None else 'Unknown'}")
                    
                    else:
                        st.error("Prediction failed. Please check your inputs and try again.")
                        
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.info("Please try again or check if the model is properly trained.")
        
        else:
            st.info("Train the model first to make predictions.")
    
    with tab4:
        st.header("Scenario Analysis")
        
        if dashboard.model_trained and dashboard.sample_data is not None:
            st.subheader("Batch Scenario Testing")
            
            scenario_type = st.selectbox(
                "Select Scenario Type:",
                ["Hour of Day Impact", "Location Comparison", "EV vs Conventional", "Driver Efficiency"]
            )
            
            try:
                if scenario_type == "Hour of Day Impact":
                    st.subheader("Wait Time Throughout the Day")
                    
                    # Create predictions for each hour
                    base_scenario = {
                        'location': 'Downtown',
                        'driver': dashboard.generator.drivers[0] if dashboard.generator.drivers else 'Driver_001',
                        'checklist_length': 8,
                        'is_ev': False,
                        'needs_fuel': True,
                        'needs_charge': False,
                        'driver_needs_lunch': False
                    }
                    
                    hourly_predictions = []
                    for hour in range(24):
                        X_pred = dashboard.handle_unknown_values(
                            location=base_scenario['location'],
                            driver=base_scenario['driver'],
                            hour=hour,
                            checklist_length=base_scenario['checklist_length'],
                            is_ev=base_scenario['is_ev'],
                            needs_fuel=base_scenario['needs_fuel'],
                            needs_charge=base_scenario['needs_charge'],
                            driver_needs_lunch=base_scenario['driver_needs_lunch']
                        )
                        
                        pred_mean, pred_std = dashboard.make_predictions(X_pred)
                        
                        if pred_mean is not None and pred_std is not None:
                            hourly_predictions.append({
                                'hour': hour,
                                'predicted_wait': pred_mean[0],
                                'uncertainty': pred_std[0]
                            })
                    
                    if hourly_predictions:
                        hourly_df = pd.DataFrame(hourly_predictions)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=hourly_df['hour'],
                            y=hourly_df['predicted_wait'],
                            mode='lines+markers',
                            name='Predicted Wait Time',
                            error_y=dict(
                                type='data',
                                array=hourly_df['uncertainty'],
                                visible=True
                            ),
                            line=dict(color='blue', width=3),
                            marker=dict(size=8)
                        ))
                        
                        fig.update_layout(
                            title="Predicted Wait Time Throughout the Day",
                            xaxis_title="Hour of Day",
                            yaxis_title="Wait Time (minutes)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(hourly_df.round(2))
                    else:
                        st.error("Failed to generate hourly predictions")
                
                elif scenario_type == "Location Comparison":
                    st.subheader("Wait Times by Location")
                    
                    location_predictions = []
                    base_scenario = {
                        'driver': dashboard.generator.drivers[0] if dashboard.generator.drivers else 'Driver_001',
                        'hour': 14,  # 2 PM
                        'checklist_length': 8,
                        'is_ev': False,
                        'needs_fuel': True,
                        'needs_charge': False,
                        'driver_needs_lunch': False
                    }
                    
                    for location in dashboard.generator.locations:
                        X_pred = dashboard.handle_unknown_values(
                            location=location,
                            driver=base_scenario['driver'],
                            hour=base_scenario['hour'],
                            checklist_length=base_scenario['checklist_length'],
                            is_ev=base_scenario['is_ev'],
                            needs_fuel=base_scenario['needs_fuel'],
                            needs_charge=base_scenario['needs_charge'],
                            driver_needs_lunch=base_scenario['driver_needs_lunch']
                        )
                        
                        pred_mean, pred_std = dashboard.make_predictions(X_pred)
                        
                        if pred_mean is not None and pred_std is not None:
                            location_predictions.append({
                                'location': location,
                                'predicted_wait': pred_mean[0],
                                'uncertainty': pred_std[0]
                            })
                    
                    if location_predictions:
                        location_df = pd.DataFrame(location_predictions)
                        
                        fig = px.bar(
                            location_df,
                            x='location',
                            y='predicted_wait',
                            error_y='uncertainty',
                            title="Average Wait Time by Location",
                            labels={'predicted_wait': 'Wait Time (minutes)', 'location': 'Location'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(location_df.round(2))
                    else:
                        st.error("Failed to generate location predictions")
                
                elif scenario_type == "EV vs Conventional":
                    st.subheader("Electric vs Conventional Vehicles")
                    
                    vehicle_predictions = []
                    base_scenario = {
                        'location': 'Downtown',
                        'driver': dashboard.generator.drivers[0] if dashboard.generator.drivers else 'Driver_001',
                        'hour': 14,
                        'checklist_length': 8,
                        'driver_needs_lunch': False
                    }
                    
                    # Conventional vehicle scenarios
                    for needs_fuel in [True, False]:
                        X_pred = dashboard.handle_unknown_values(
                            location=base_scenario['location'],
                            driver=base_scenario['driver'],
                            hour=base_scenario['hour'],
                            checklist_length=base_scenario['checklist_length'],
                            is_ev=False,
                            needs_fuel=needs_fuel,
                            needs_charge=False,
                            driver_needs_lunch=base_scenario['driver_needs_lunch']
                        )
                        
                        pred_mean, pred_std = dashboard.make_predictions(X_pred)
                        
                        if pred_mean is not None and pred_std is not None:
                            vehicle_predictions.append({
                                'vehicle_type': f"Conventional ({'Needs Fuel' if needs_fuel else 'No Fuel'})",
                                'predicted_wait': pred_mean[0],
                                'uncertainty': pred_std[0]
                            })
                    
                    # Electric vehicle scenarios
                    for needs_charge in [True, False]:
                        X_pred = dashboard.handle_unknown_values(
                            location=base_scenario['location'],
                            driver=base_scenario['driver'],
                            hour=base_scenario['hour'],
                            checklist_length=base_scenario['checklist_length'],
                            is_ev=True,
                            needs_fuel=False,
                            needs_charge=needs_charge,
                            driver_needs_lunch=base_scenario['driver_needs_lunch']
                        )
                        
                        pred_mean, pred_std = dashboard.make_predictions(X_pred)
                        
                        if pred_mean is not None and pred_std is not None:
                            vehicle_predictions.append({
                                'vehicle_type': f"Electric ({'Needs Charge' if needs_charge else 'No Charge'})",
                                'predicted_wait': pred_mean[0],
                                'uncertainty': pred_std[0]
                            })
                    
                    if vehicle_predictions:
                        vehicle_df = pd.DataFrame(vehicle_predictions)
                        
                        fig = px.bar(
                            vehicle_df,
                            x='vehicle_type',
                            y='predicted_wait',
                            error_y='uncertainty',
                            title="Wait Time by Vehicle Type and Service Needs",
                            labels={'predicted_wait': 'Wait Time (minutes)', 'vehicle_type': 'Vehicle Type'}
                        )
                        fig.update_layout(height=400)
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(vehicle_df.round(2))
                    else:
                        st.error("Failed to generate vehicle comparison predictions")
                
                elif scenario_type == "Driver Efficiency":
                    st.subheader("Driver Performance Comparison")
                    
                    # Select a sample of drivers for comparison
                    sample_drivers = dashboard.generator.drivers[:5]  # First 5 drivers
                    driver_predictions = []
                    
                    base_scenario = {
                        'location': 'Downtown',
                        'hour': 14,
                        'checklist_length': 8,
                        'is_ev': False,
                        'needs_fuel': True,
                        'needs_charge': False,
                        'driver_needs_lunch': False
                    }
                    
                    for driver in sample_drivers:
                        X_pred = dashboard.handle_unknown_values(
                            location=base_scenario['location'],
                            driver=driver,
                            hour=base_scenario['hour'],
                            checklist_length=base_scenario['checklist_length'],
                            is_ev=base_scenario['is_ev'],
                            needs_fuel=base_scenario['needs_fuel'],
                            needs_charge=base_scenario['needs_charge'],
                            driver_needs_lunch=base_scenario['driver_needs_lunch']
                        )
                        
                        pred_mean, pred_std = dashboard.make_predictions(X_pred)
                        
                        if pred_mean is not None and pred_std is not None:
                            driver_predictions.append({
                                'driver': driver,
                                'predicted_wait': pred_mean[0],
                                'uncertainty': pred_std[0]
                            })
                    
                    if driver_predictions:
                        driver_df = pd.DataFrame(driver_predictions)
                        
                        fig = px.bar(
                            driver_df,
                            x='driver',
                            y='predicted_wait',
                            error_y='uncertainty',
                            title="Wait Time by Driver (Same Conditions)",
                            labels={'predicted_wait': 'Wait Time (minutes)', 'driver': 'Driver'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(driver_df.round(2))
                        
                        st.info("💡 **Note**: Driver efficiency is learned from the training data. Differences reflect historical performance patterns.")
                    else:
                        st.error("Failed to generate driver comparison predictions")
            
            except Exception as e:
                st.error(f"Scenario analysis error: {str(e)}")
                st.info("Please ensure the model is properly trained and try again.")
        
        else:
            st.info("Train the model and generate data to perform scenario analysis.")
    
    with tab5:
        st.header("Model Tuning & Optimization")
        
        if dashboard.sample_data is not None:
            st.subheader("Hyperparameter Tuning")
            
            st.write("Configure MCMC parameters for model training:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_samples_options = [500, 1000, 1500, 2000]
                selected_samples = st.selectbox("Number of Samples", num_samples_options, index=1)
            
            with col2:
                warmup_options = [250, 500, 750, 1000]
                selected_warmup = st.selectbox("Warmup Steps", warmup_options, index=1)
            
            with col3:
                chain_options = [1, 2, 4, 6]
                selected_chains = st.selectbox("Number of Chains", chain_options, index=0)
            
            st.info("💡 **Chain Selection Guide:**\n"
                   "- **1 chain**: Fastest, most stable for Streamlit\n"
                   "- **2-4 chains**: Better convergence diagnostics\n" 
                   "- **6 chains**: Maximum parallelism (auto-capped at 8 cores to preserve web interface)")
            
            if st.button("Train with Custom Parameters"):
                if selected_chains == 1:
                    st.info("ℹ️ Using single-chain mode for maximum Streamlit compatibility.")
                    with st.spinner(f"Training model with {selected_samples} samples, {selected_warmup} warmup steps..."):
                        try:
                            X, y, encodings = dashboard.generator.get_feature_encodings(dashboard.sample_data)
                            
                            n_locations = len(dashboard.generator.locations)
                            n_drivers = len(dashboard.generator.drivers)
                            dashboard.model = BayesianWaitTimeModel(n_locations, n_drivers, device='auto')
                            
                            results = dashboard.model.fit_streamlit_safe(
                                X, y,
                                num_samples=selected_samples,
                                warmup_steps=selected_warmup
                            )
                            
                            dashboard.model_trained = True
                            st.session_state.model_results = results
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
                else:
                    st.info(f"ℹ️ Using multi-chain mode ({selected_chains} chains) with CPU capping for web interface stability.")
                    with st.spinner(f"Training model with {selected_chains} chains, {selected_samples} samples, {selected_warmup} warmup steps..."):
                        try:
                            X, y, encodings = dashboard.generator.get_feature_encodings(dashboard.sample_data)
                            
                            n_locations = len(dashboard.generator.locations)
                            n_drivers = len(dashboard.generator.drivers)
                            dashboard.model = BayesianWaitTimeModel(n_locations, n_drivers, device='auto')
                            
                            # Use the improved fit method with CPU capping
                            results = dashboard.model.fit(
                                X, y,
                                num_samples=selected_samples,
                                warmup_steps=selected_warmup,
                                num_chains=selected_chains,
                                disable_progbar=True  # Disable progress bar for Streamlit
                            )
                            
                            dashboard.model_trained = True
                            st.session_state.model_results = results
                            st.session_state.encodings = encodings
                            st.success("Model trained with custom parameters!")
                            
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
                            st.info("💡 Try reducing the number of samples or warmup steps.")
            
            st.subheader("Performance Metrics")
            
            if dashboard.model_trained:
                try:
                    # Calculate and display performance metrics
                    X, y, _ = dashboard.generator.get_feature_encodings(dashboard.sample_data)
                    
                    if len(X) < 10:
                        st.warning("Dataset too small for meaningful validation. Generate more data.")
                        return
                    
                    # Split data for validation
                    n_train = int(0.8 * len(X))
                    X_train, X_val = X[:n_train], X[n_train:]
                    y_train, y_val = y[:n_train], y[n_train:]
                    
                    if len(X_val) == 0:
                        st.warning("No validation data available. Increase dataset size.")
                        return
                    
                    # Make predictions on validation set
                    y_pred, y_std = dashboard.make_predictions(X_val)
                    
                    if y_pred is not None and len(y_pred) > 0:
                        # Ensure arrays have same length
                        min_len = min(len(y_val), len(y_pred))
                        y_val = y_val[:min_len]
                        y_pred = y_pred[:min_len]
                        
                        mse = np.mean((y_val - y_pred) ** 2)
                        mae = np.mean(np.abs(y_val - y_pred))
                        rmse = np.sqrt(mse)
                        
                        # R-squared
                        ss_res = np.sum((y_val - y_pred) ** 2)
                        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MSE", f"{mse:.2f}")
                        with col2:
                            st.metric("MAE", f"{mae:.2f}")
                        with col3:
                            st.metric("RMSE", f"{rmse:.2f}")
                        with col4:
                            st.metric("R²", f"{r2:.3f}")
                        
                        if len(y_val) > 1 and len(y_pred) > 1:
                            # Prediction vs actual plot
                            fig = px.scatter(
                                x=y_val, y=y_pred,
                                title="Predicted vs Actual Wait Times",
                                labels={'x': 'Actual Wait Time (minutes)', 'y': 'Predicted Wait Time (minutes)'}
                            )
                            
                            # Add perfect prediction line
                            min_val = min(min(y_val), min(y_pred))
                            max_val = max(max(y_val), max(y_pred))
                            fig.add_shape(
                                type='line',
                                line=dict(dash='dash', color='red'),
                                x0=min_val, y0=min_val,
                                x1=max_val, y1=max_val
                            )
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Performance interpretation
                            if r2 > 0.7:
                                st.success("✅ Good model performance (R² > 0.7)")
                            elif r2 > 0.5:
                                st.info("📊 Moderate model performance (R² > 0.5)")
                            else:
                                st.warning("⚠️ Model performance could be improved (R² < 0.5)")
                                st.info("💡 Try: More training data, different hyperparameters, or model architecture")
                        else:
                            st.error("Insufficient data for visualization")
                    else:
                        st.error("Failed to generate predictions for validation")
                        
                except Exception as e:
                    st.error(f"Performance evaluation error: {str(e)}")
                    st.info("This might be due to insufficient training data or model issues.")
            else:
                st.info("Train a model first to see performance metrics.")
        
        else:
            st.info("Generate data first to access model tuning options.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this dashboard:** This interactive dashboard demonstrates Bayesian MCMC modeling for 
    car collection wait time prediction. The model incorporates uncertainty quantification and 
    provides credible intervals for predictions.
    
    **Technical Stack:** PyTorch, Pyro, Streamlit, Plotly
    """)


if __name__ == "__main__":
    main()