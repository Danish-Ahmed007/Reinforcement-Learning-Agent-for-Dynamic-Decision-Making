"""Utility functions for plotting, logging, and data handling."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List
import json
from datetime import datetime


def create_directories():
    """Create necessary directories."""
    directories = ['models', 'results', 'plots', 'tensorboard_logs', 'videos']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Created necessary directories")


def plot_training_results(rewards: List[float], episode_lengths: List[int], 
                          save_path: str = 'plots/training_results.png'):
    """Plot training rewards and episode lengths."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(rewards, alpha=0.6, label='Episode Reward')
    if len(rewards) > 10:
        window = min(100, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 
                color='red', linewidth=2, label=f'Moving Average ({window} episodes)')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(episode_lengths, alpha=0.6, color='green', label='Episode Length')
    if len(episode_lengths) > 10:
        window = min(100, len(episode_lengths) // 10)
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), moving_avg, 
                color='red', linewidth=2, label=f'Moving Average ({window} episodes)')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training results plot saved to {save_path}")


def plot_evaluation_results(eval_rewards: List[float], 
                           save_path: str = 'plots/evaluation_results.png'):
    """Plot evaluation rewards."""
    plt.figure(figsize=(12, 6))
    episodes = range(1, len(eval_rewards) + 1)
    
    plt.bar(episodes, eval_rewards, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.axhline(y=np.mean(eval_rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(eval_rewards):.2f}')
    plt.axhline(y=200, color='green', linestyle='--', 
                linewidth=2, label='Success Threshold (200)')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Evaluation Results')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Evaluation results plot saved to {save_path}")


def save_results(results_dict: dict, filename: str = 'results/training_results.json'):
    """Save results to JSON file."""
    results_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"Results saved to {filename}")


def print_statistics(rewards: List[float], episode_lengths: List[int]):
    """Print performance statistics."""
    print("\n" + "="*60)
    print("PERFORMANCE STATISTICS")
    print("="*60)
    print(f"Total Episodes: {len(rewards)}")
    print(f"\nRewards:")
    print(f"  Mean:   {np.mean(rewards):.2f}")
    print(f"  Std:    {np.std(rewards):.2f}")
    print(f"  Min:    {np.min(rewards):.2f}")
    print(f"  Max:    {np.max(rewards):.2f}")
    print(f"  Median: {np.median(rewards):.2f}")
    
    print(f"\nEpisode Lengths:")
    print(f"  Mean:   {np.mean(episode_lengths):.2f}")
    print(f"  Std:    {np.std(episode_lengths):.2f}")
    print(f"  Min:    {int(np.min(episode_lengths))}")
    print(f"  Max:    {int(np.max(episode_lengths))}")
    
    success_rate = sum(1 for r in rewards if r >= 200) / len(rewards) * 100
    print(f"\nSuccess Rate (reward >= 200): {success_rate:.1f}%")
    print("="*60 + "\n")


def calculate_moving_average(data: List[float], window: int = 100) -> np.ndarray:
    """Calculate moving average."""
    if len(data) < window:
        window = len(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')
