#!/usr/bin/env python3
"""
Script to run RL vs MCMC comparison for car collection wait time prediction.
This demonstrates whether RL approaches improve on statistical methods.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import argparse
import pickle
from datetime import datetime

from data_generator import CarCollectionDataGenerator
from mcmc_model import BayesianWaitTimeModel
from rl_comparison import RLBenchmark
from wandb_logger import WandBLogger


def main():
    parser = argparse.ArgumentParser(description='Compare RL vs MCMC for booking decisions')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained MCMC model pickle file')
    parser.add_argument('--rl_episodes', type=int, default=1000, 
                       help='Number of episodes to train RL agent')
    parser.add_argument('--eval_episodes', type=int, default=500, 
                       help='Number of episodes for evaluation')
    parser.add_argument('--output_dir', type=str, default='outputs', 
                       help='Output directory for results')
    parser.add_argument('--use_wandb', action='store_true', 
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='rl-vs-mcmc-comparison',
                       help='WandB project name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    
    print("🤖 RL vs MCMC Comparison - Car Collection Booking")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'rl_comparison'), exist_ok=True)
    
    # Initialize WandB if requested
    wandb_logger = None
    if args.use_wandb:
        print("1. Initializing WandB logging...")
        config = {
            'rl_episodes': args.rl_episodes,
            'eval_episodes': args.eval_episodes,
            'seed': args.seed,
            'model_path': args.model_path
        }
        
        wandb_logger = WandBLogger(project_name=args.wandb_project, config=config)
        run = wandb_logger.start_run(run_name=f"rl_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"   - WandB run started: {run.name}")
    
    # Load trained MCMC model
    print("2. Loading trained MCMC model...")
    try:
        with open(args.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        mcmc_model = model_data['model']
        print(f"   - Model loaded from: {args.model_path}")
        print(f"   - Model has {len(mcmc_model.samples) if mcmc_model.samples else 0} parameter samples")
        
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {args.model_path}")
        print("   Please train an MCMC model first using main_training.py")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize components
    print("3. Initializing comparison framework...")
    data_generator = CarCollectionDataGenerator(seed=args.seed)
    benchmark = RLBenchmark(mcmc_model, data_generator)
    print("   - RL benchmark environment initialized")
    
    # Run comparison
    print(f"4. Running RL vs MCMC comparison...")
    print(f"   - RL training episodes: {args.rl_episodes}")
    print(f"   - Evaluation episodes: {args.eval_episodes}")
    print("   - This may take several minutes...")
    
    results = benchmark.compare_methods(
        rl_episodes=args.rl_episodes,
        eval_episodes=args.eval_episodes
    )
    
    # Analyze results
    print(f"\n5. Analyzing results...")
    
    improvement_percent = (results['improvement'] / abs(results['mcmc_mean'])) * 100
    
    print(f"   📊 Performance Summary:")
    print(f"   ├── RL Agent Performance: {results['rl_mean']:.2f} ± {results['rl_std']:.2f}")
    print(f"   ├── MCMC Policy Performance: {results['mcmc_mean']:.2f} ± {results['mcmc_std']:.2f}")
    print(f"   ├── Absolute Improvement: {results['improvement']:.2f}")
    print(f"   └── Relative Improvement: {improvement_percent:.1f}%")
    
    # Statistical significance test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(results['rl_eval_rewards'], results['mcmc_rewards'])
    
    print(f"\n   📈 Statistical Analysis:")
    print(f"   ├── T-statistic: {t_stat:.3f}")
    print(f"   ├── P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        winner = "RL Agent" if results['rl_mean'] > results['mcmc_mean'] else "MCMC Policy"
        print(f"   └── 🏆 {winner} is significantly better (p < 0.05)")
    else:
        print(f"   └── 🤝 No significant difference detected (p ≥ 0.05)")
    
    # Create visualizations
    print(f"6. Creating visualizations...")
    fig = benchmark.plot_comparison(
        results, 
        save_path=os.path.join(args.output_dir, 'rl_comparison', 'comparison_results.png')
    )
    print("   - Comparison plots saved")
    
    # Log to WandB
    if wandb_logger:
        print("   - Logging results to WandB...")
        
        metrics = {
            'rl_mean_reward': results['rl_mean'],
            'rl_std_reward': results['rl_std'],
            'mcmc_mean_reward': results['mcmc_mean'],
            'mcmc_std_reward': results['mcmc_std'],
            'improvement_absolute': results['improvement'],
            'improvement_percent': improvement_percent,
            't_statistic': t_stat,
            'p_value': p_value
        }
        
        wandb_logger.run.log(metrics)
        
        # Log comparison plot
        import wandb
        wandb_logger.run.log({"rl_vs_mcmc_comparison": wandb.Image(fig)})
        
        # Log reward distributions
        import plotly.figure_factory as ff
        hist_data = [results['rl_eval_rewards'], results['mcmc_rewards']]
        group_labels = ['RL Agent', 'MCMC Policy']
        
        fig_dist = ff.create_distplot(hist_data, group_labels, bin_size=2)
        fig_dist.update_layout(title="Reward Distributions Comparison")
        wandb_logger.run.log({"reward_distributions": wandb.Plotly(fig_dist)})
    
    # Save detailed results
    print("7. Saving results...")
    
    results_summary = {
        'experiment_config': {
            'rl_episodes': args.rl_episodes,
            'eval_episodes': args.eval_episodes,
            'seed': args.seed,
            'model_path': args.model_path,
            'timestamp': datetime.now().isoformat()
        },
        'performance_metrics': {
            'rl_mean': results['rl_mean'],
            'rl_std': results['rl_std'],
            'mcmc_mean': results['mcmc_mean'],
            'mcmc_std': results['mcmc_std'],
            'improvement_absolute': results['improvement'],
            'improvement_percent': improvement_percent
        },
        'statistical_analysis': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'better_method': 'rl' if results['rl_mean'] > results['mcmc_mean'] else 'mcmc'
        },
        'raw_results': {
            'rl_eval_rewards': results['rl_eval_rewards'],
            'mcmc_rewards': results['mcmc_rewards']
        }
    }
    
    # Save to JSON
    import json
    results_path = os.path.join(args.output_dir, 'rl_comparison', 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"   - Results saved to: {results_path}")
    
    # Save raw results for further analysis
    import pandas as pd
    
    eval_df = pd.DataFrame({
        'episode': list(range(len(results['rl_eval_rewards']))),
        'rl_reward': results['rl_eval_rewards'],
        'mcmc_reward': results['mcmc_rewards']
    })
    
    eval_path = os.path.join(args.output_dir, 'rl_comparison', 'evaluation_results.csv')
    eval_df.to_csv(eval_path, index=False)
    print(f"   - Evaluation data saved to: {eval_path}")
    
    # Conclusions
    print(f"\n{'='*60}")
    print("🎯 Conclusions:")
    
    if results['improvement'] > 0:
        print(f"✅ RL Agent outperformed MCMC Policy by {improvement_percent:.1f}%")
        if p_value < 0.05:
            print("✅ The improvement is statistically significant")
        else:
            print("⚠️  The improvement is not statistically significant")
    else:
        print(f"❌ MCMC Policy outperformed RL Agent by {abs(improvement_percent):.1f}%")
        if p_value < 0.05:
            print("✅ MCMC's superior performance is statistically significant")
        else:
            print("⚠️  The difference is not statistically significant")
    
    print(f"\n💡 Insights:")
    
    if abs(improvement_percent) < 5:
        print("   📊 Both methods perform similarly - choice may depend on other factors")
        print("   📊 MCMC provides uncertainty quantification, RL adapts to environment")
    elif results['improvement'] > 0:
        print("   🤖 RL shows promise for dynamic booking decisions")
        print("   🤖 Consider hybrid approaches combining both methods")
    else:
        print("   📈 Statistical MCMC approach is more robust for this problem")
        print("   📈 RL may need more sophisticated features or training")
    
    # Finish WandB
    if wandb_logger:
        print("\n   - Finishing WandB run...")
        wandb_logger.finish_run()
    
    print(f"\n📁 All results saved to: {args.output_dir}/rl_comparison/")
    print("🎉 RL vs MCMC comparison completed!")


if __name__ == "__main__":
    main()