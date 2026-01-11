"""Compare trained models and generate comparison plots."""

import matplotlib
matplotlib.use('Agg')

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import os
import json

import config


def compare_models(model_paths, labels, n_episodes=50):
    """Compare multiple trained models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    all_results = {}
    
    for model_path, label in zip(model_paths, labels):
        print(f"\nEvaluating {label}...")
        
        if not os.path.exists(model_path + ".zip"):
            print(f"  Model not found: {model_path}")
            continue
        
        model = DQN.load(model_path)
        env = gym.make(config.ENV_NAME)
        
        episode_rewards = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        env.close()
        
        all_results[label] = {
            'rewards': episode_rewards,
            'mean': np.mean(episode_rewards),
            'std': np.std(episode_rewards),
            'min': np.min(episode_rewards),
            'max': np.max(episode_rewards),
            'success_rate': sum(1 for r in episode_rewards if r >= 200) / n_episodes * 100
        }
        
        print(f"  Mean Reward: {all_results[label]['mean']:.2f} +/- {all_results[label]['std']:.2f}")
        print(f"  Success Rate: {all_results[label]['success_rate']:.1f}%")
    
    return all_results


def plot_comparison(results, save_path='plots/model_comparison.png'):
    """Plot comparison of models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    labels = list(results.keys())
    means = [results[label]['mean'] for label in labels]
    stds = [results[label]['std'] for label in labels]
    success_rates = [results[label]['success_rate'] for label in labels]
    
    axes[0, 0].bar(labels, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[0, 0].axhline(y=200, color='green', linestyle='--', label='Success Threshold')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].set_title('Mean Rewards Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].bar(labels, success_rates, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].set_title('Success Rate Comparison')
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    reward_data = [results[label]['rewards'] for label in labels]
    axes[1, 0].boxplot(reward_data, labels=labels)
    axes[1, 0].axhline(y=200, color='green', linestyle='--', label='Success Threshold')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    parts = axes[1, 1].violinplot(reward_data, positions=range(len(labels)), 
                                   showmeans=True, showmedians=True)
    axes[1, 1].axhline(y=200, color='green', linestyle='--', label='Success Threshold')
    axes[1, 1].set_xticks(range(len(labels)))
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].set_title('Reward Distribution (Violin)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nComparison plot saved to {save_path}")


def main():
    """Main comparison function."""
    model_paths = [
        os.path.join(config.MODEL_SAVE_PATH, "best_model"),
        os.path.join(config.MODEL_SAVE_PATH, "final_model"),
    ]
    
    labels = ["Best Model", "Final Model"]
    
    existing_models = []
    existing_labels = []
    
    for path, label in zip(model_paths, labels):
        if os.path.exists(path + ".zip"):
            existing_models.append(path)
            existing_labels.append(label)
    
    if len(existing_models) == 0:
        print("\nNo trained models found.")
        print("Please train the agent first by running: python train.py")
        return
    
    if len(existing_models) == 1:
        print("\nOnly one model found. Need at least 2 models for comparison.")
        print(f"Found: {existing_labels[0]}")
        return
    
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    results = compare_models(existing_models, existing_labels, n_episodes=50)
    
    plot_comparison(results)
    
    results_to_save = {label: {k: v for k, v in data.items() if k != 'rewards'} 
                      for label, data in results.items()}
    
    with open('results/model_comparison.json', 'w') as f:
        json.dump(results_to_save, f, indent=4)
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    for label in existing_labels:
        if label in results:
            print(f"\n{label}:")
            print(f"  Mean Reward: {results[label]['mean']:.2f} +/- {results[label]['std']:.2f}")
            print(f"  Success Rate: {results[label]['success_rate']:.1f}%")
            print(f"  Best Episode: {results[label]['max']:.2f}")
            print(f"  Worst Episode: {results[label]['min']:.2f}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
