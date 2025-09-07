import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class BookingState:
    """Represents the state of a booking decision scenario."""
    current_time: int  # Hour of day
    location: int      # Location index
    driver_available: List[int]  # Available driver indices
    estimated_wait_times: List[float]  # Estimated wait times for each driver
    public_transport_cost: float  # Cost of public transport option
    public_transport_time: float   # Time for public transport
    booking_deadline: int  # Hours until must make decision


@dataclass
class Action:
    """Represents an action in the booking scenario."""
    action_type: str  # 'book_driver', 'public_transport', 'wait'
    driver_id: Optional[int] = None


@dataclass
class Experience:
    """Experience tuple for replay buffer."""
    state: BookingState
    action: Action
    reward: float
    next_state: BookingState
    done: bool


class BookingEnvironment:
    """
    Environment for simulating car booking decisions.
    Agents must decide between booking a driver, taking public transport, or waiting.
    """
    
    def __init__(self, n_locations: int = 6, n_drivers: int = 20, max_episodes: int = 100):
        self.n_locations = n_locations
        self.n_drivers = n_drivers
        self.max_episodes = max_episodes
        self.current_episode = 0
        
        # Environment parameters
        self.public_transport_base_cost = 15.0
        self.public_transport_base_time = 45.0
        self.waiting_penalty = 2.0  # Penalty per hour for waiting
        self.booking_reward_base = 50.0
        
        self.reset()
    
    def reset(self) -> BookingState:
        """Reset environment to initial state."""
        self.current_episode += 1
        self.current_time = np.random.randint(6, 22)  # 6 AM to 10 PM
        self.location = np.random.randint(0, self.n_locations)
        
        # Random subset of available drivers
        n_available = np.random.randint(3, min(8, self.n_drivers))
        self.available_drivers = np.random.choice(self.n_drivers, n_available, replace=False)
        
        # Simulate estimated wait times (with some noise/uncertainty)
        base_wait_times = np.random.exponential(20, len(self.available_drivers))
        time_factor = 1.4 if (7 <= self.current_time <= 9 or 17 <= self.current_time <= 19) else 1.0
        self.estimated_wait_times = base_wait_times * time_factor
        
        # Public transport options
        self.public_transport_cost = self.public_transport_base_cost * (1 + np.random.normal(0, 0.2))
        self.public_transport_time = self.public_transport_base_time * (1 + np.random.normal(0, 0.3))
        
        # Booking deadline (hours from now)
        self.booking_deadline = np.random.randint(1, 6)
        self.steps_taken = 0
        
        return self._get_state()
    
    def _get_state(self) -> BookingState:
        """Get current state."""
        return BookingState(
            current_time=self.current_time,
            location=self.location,
            driver_available=list(self.available_drivers),
            estimated_wait_times=list(self.estimated_wait_times),
            public_transport_cost=self.public_transport_cost,
            public_transport_time=self.public_transport_time,
            booking_deadline=self.booking_deadline
        )
    
    def step(self, action: Action) -> Tuple[BookingState, float, bool, Dict]:
        """Execute action and return next state, reward, done, info."""
        reward = 0
        done = False
        info = {}
        
        if action.action_type == 'book_driver':
            if action.driver_id in self.available_drivers:
                driver_idx = list(self.available_drivers).index(action.driver_id)
                actual_wait_time = self.estimated_wait_times[driver_idx] * (1 + np.random.normal(0, 0.2))
                
                # Reward based on efficiency (lower wait time = higher reward)
                reward = self.booking_reward_base - actual_wait_time * 0.5
                info['actual_wait_time'] = actual_wait_time
                info['action_taken'] = 'booked_driver'
                done = True
            else:
                # Invalid action
                reward = -10
                info['action_taken'] = 'invalid_booking'
        
        elif action.action_type == 'public_transport':
            # Fixed cost and time for public transport
            reward = self.booking_reward_base - self.public_transport_cost - self.public_transport_time * 0.2
            info['cost'] = self.public_transport_cost
            info['time'] = self.public_transport_time
            info['action_taken'] = 'public_transport'
            done = True
        
        elif action.action_type == 'wait':
            # Small penalty for waiting, but might get better options
            reward = -self.waiting_penalty
            self.steps_taken += 1
            self.booking_deadline -= 1
            
            # Simulate changes in available drivers and wait times
            if np.random.random() < 0.3:  # 30% chance of changes
                # Some drivers might become unavailable
                if len(self.available_drivers) > 1:
                    to_remove = np.random.choice(self.available_drivers, 
                                               max(1, len(self.available_drivers) // 4), 
                                               replace=False)
                    self.available_drivers = [d for d in self.available_drivers if d not in to_remove]
                    # Remove corresponding wait times
                    keep_indices = [i for i, d in enumerate(self.available_drivers) if d not in to_remove]
                    self.estimated_wait_times = [self.estimated_wait_times[i] for i in keep_indices]
                
                # New drivers might become available
                unavailable_drivers = [d for d in range(self.n_drivers) if d not in self.available_drivers]
                if unavailable_drivers and np.random.random() < 0.2:
                    new_driver = np.random.choice(unavailable_drivers)
                    self.available_drivers = np.append(self.available_drivers, new_driver)
                    new_wait_time = np.random.exponential(18)  # Slightly better wait time
                    self.estimated_wait_times.append(new_wait_time)
            
            info['action_taken'] = 'waited'
            
            # Check if deadline reached
            if self.booking_deadline <= 0:
                # Forced to take public transport with penalty
                reward += self.booking_reward_base - self.public_transport_cost * 1.5 - 20  # Deadline penalty
                info['forced_public_transport'] = True
                done = True
        
        return self._get_state(), reward, done, info


class DQNAgent:
    """Deep Q-Network agent for booking decisions."""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural network
        self.q_network = self._build_model().to(self.device)
        self.target_network = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
    
    def _build_model(self) -> nn.Module:
        """Build the neural network model."""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def _state_to_vector(self, state: BookingState) -> np.ndarray:
        """Convert state to feature vector."""
        features = [
            state.current_time / 24.0,  # Normalized hour
            state.location / 6.0,       # Normalized location
            len(state.driver_available) / 20.0,  # Normalized driver count
            np.mean(state.estimated_wait_times) / 60.0,  # Normalized avg wait time
            np.min(state.estimated_wait_times) / 60.0,   # Normalized min wait time
            state.public_transport_cost / 50.0,  # Normalized PT cost
            state.public_transport_time / 120.0, # Normalized PT time
            state.booking_deadline / 6.0,        # Normalized deadline
        ]
        
        return np.array(features, dtype=np.float32)
    
    def remember(self, experience: Experience):
        """Store experience in replay buffer."""
        self.memory.append(experience)
    
    def act(self, state: BookingState) -> Action:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            # Random action
            action_type = np.random.choice(['book_driver', 'public_transport', 'wait'])
            if action_type == 'book_driver' and state.driver_available:
                driver_id = np.random.choice(state.driver_available)
                return Action(action_type, driver_id)
            else:
                return Action(action_type)
        
        # Get Q-values from network
        state_vector = torch.FloatTensor(self._state_to_vector(state)).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_vector)
        
        # Choose best action
        action_idx = q_values.argmax().item()
        
        if action_idx == 0:  # book_driver
            if state.driver_available:
                # Choose driver with minimum estimated wait time
                best_driver_idx = np.argmin(state.estimated_wait_times)
                driver_id = state.driver_available[best_driver_idx]
                return Action('book_driver', driver_id)
            else:
                return Action('wait')
        elif action_idx == 1:  # public_transport
            return Action('public_transport')
        else:  # wait
            return Action('wait')
    
    def replay(self, batch_size: int = 32):
        """Train the model on a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([self._state_to_vector(e.state) for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([self._state_to_vector(e.next_state) for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states)
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states)
        target_q_values = rewards + (0.99 * next_q_values.max(1)[0] * ~dones)
        
        # Convert actions to indices
        action_indices = []
        for e in batch:
            if e.action.action_type == 'book_driver':
                action_indices.append(0)
            elif e.action.action_type == 'public_transport':
                action_indices.append(1)
            else:
                action_indices.append(2)
        
        action_indices = torch.LongTensor(action_indices).to(self.device)
        
        # Compute loss
        predicted_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(predicted_q_values, target_q_values.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())


class RLBenchmark:
    """Benchmark RL approach against MCMC statistical method."""
    
    def __init__(self, mcmc_model, data_generator):
        self.mcmc_model = mcmc_model
        self.data_generator = data_generator
        self.env = BookingEnvironment()
    
    def train_rl_agent(self, episodes: int = 1000) -> Tuple[DQNAgent, List[float]]:
        """Train DQN agent and return training rewards."""
        agent = DQNAgent(state_size=8, action_size=3)
        rewards = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            while True:
                action = agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                experience = Experience(state, action, reward, next_state, done)
                agent.remember(experience)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            rewards.append(total_reward)
            
            # Train agent
            if len(agent.memory) > 32:
                agent.replay(32)
            
            # Update target network periodically
            if episode % 10 == 0:
                agent.update_target_network()
            
            if episode % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        return agent, rewards
    
    def evaluate_mcmc_policy(self, episodes: int = 200) -> List[float]:
        """Evaluate MCMC-based policy."""
        rewards = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            while True:
                # Use MCMC model to make decision
                action = self._mcmc_decision(state)
                next_state, reward, done, info = self.env.step(action)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            rewards.append(total_reward)
        
        return rewards
    
    def _mcmc_decision(self, state: BookingState) -> Action:
        """Make decision using MCMC model predictions."""
        if not state.driver_available:
            return Action('public_transport')
        
        # Get MCMC predictions for each driver
        driver_predictions = []
        for i, driver_id in enumerate(state.driver_available):
            # Create feature vector for prediction
            X_pred = np.array([[
                state.location,           # location
                driver_id % 50,          # driver (mod to fit model)
                state.current_time,      # hour
                8,                       # default checklist_length
                0,                       # is_ev
                1,                       # needs_fuel
                0,                       # needs_charge
                0                        # driver_needs_lunch
            ]], dtype=np.float32)
            
            try:
                pred_mean, pred_std = self.mcmc_model.predict(X_pred, num_samples=50)
                driver_predictions.append((driver_id, pred_mean[0], pred_std[0]))
            except:
                # Fallback to estimated wait time if model fails
                driver_predictions.append((driver_id, state.estimated_wait_times[i], 5.0))
        
        # Decision logic based on predictions and uncertainty
        best_driver = min(driver_predictions, key=lambda x: x[1])  # Minimum predicted wait
        best_wait_time = best_driver[1]
        
        # Compare with public transport
        pt_effective_time = state.public_transport_time + state.public_transport_cost * 0.5  # Cost penalty
        
        # Decision thresholds
        if best_wait_time < pt_effective_time * 0.7:  # Driver is clearly better
            return Action('book_driver', best_driver[0])
        elif best_wait_time > pt_effective_time * 1.3:  # Public transport is clearly better
            return Action('public_transport')
        else:
            # Close call - consider waiting if deadline allows
            if state.booking_deadline > 2:
                return Action('wait')
            elif best_wait_time < pt_effective_time:
                return Action('book_driver', best_driver[0])
            else:
                return Action('public_transport')
    
    def compare_methods(self, rl_episodes: int = 1000, eval_episodes: int = 200) -> Dict:
        """Compare RL and MCMC approaches."""
        print("Training RL agent...")
        rl_agent, rl_training_rewards = self.train_rl_agent(rl_episodes)
        
        print("Evaluating RL agent...")
        rl_eval_rewards = []
        for episode in range(eval_episodes):
            state = self.env.reset()
            total_reward = 0
            
            # Disable exploration for evaluation
            original_epsilon = rl_agent.epsilon
            rl_agent.epsilon = 0
            
            while True:
                action = rl_agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    break
            
            rl_eval_rewards.append(total_reward)
            rl_agent.epsilon = original_epsilon
        
        print("Evaluating MCMC policy...")
        mcmc_rewards = self.evaluate_mcmc_policy(eval_episodes)
        
        # Results
        results = {
            'rl_training_rewards': rl_training_rewards,
            'rl_eval_rewards': rl_eval_rewards,
            'mcmc_rewards': mcmc_rewards,
            'rl_mean': np.mean(rl_eval_rewards),
            'rl_std': np.std(rl_eval_rewards),
            'mcmc_mean': np.mean(mcmc_rewards),
            'mcmc_std': np.std(mcmc_rewards),
            'improvement': np.mean(rl_eval_rewards) - np.mean(mcmc_rewards)
        }
        
        print(f"\nResults:")
        print(f"RL Agent: {results['rl_mean']:.2f} ± {results['rl_std']:.2f}")
        print(f"MCMC Policy: {results['mcmc_mean']:.2f} ± {results['mcmc_std']:.2f}")
        print(f"Improvement: {results['improvement']:.2f}")
        
        return results
    
    def plot_comparison(self, results: Dict, save_path: str = None):
        """Plot comparison results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training curve
        axes[0, 0].plot(results['rl_training_rewards'])
        axes[0, 0].set_title('RL Training Progress')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Evaluation comparison
        axes[0, 1].hist(results['rl_eval_rewards'], bins=30, alpha=0.7, label='RL Agent')
        axes[0, 1].hist(results['mcmc_rewards'], bins=30, alpha=0.7, label='MCMC Policy')
        axes[0, 1].set_title('Reward Distributions')
        axes[0, 1].set_xlabel('Total Reward')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[1, 0].boxplot([results['rl_eval_rewards'], results['mcmc_rewards']], 
                          labels=['RL Agent', 'MCMC Policy'])
        axes[1, 0].set_title('Performance Comparison')
        axes[1, 0].set_ylabel('Total Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Running average
        window = 50
        rl_running_avg = pd.Series(results['rl_eval_rewards']).rolling(window=window).mean()
        mcmc_running_avg = pd.Series(results['mcmc_rewards']).rolling(window=window).mean()
        
        axes[1, 1].plot(rl_running_avg, label=f'RL Agent (avg: {results["rl_mean"]:.2f})')
        axes[1, 1].plot(mcmc_running_avg, label=f'MCMC Policy (avg: {results["mcmc_mean"]:.2f})')
        axes[1, 1].set_title(f'Running Average (window={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('RL vs MCMC Comparison Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig