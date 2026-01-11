"""Evaluation script for trained Lunar Lander DQN Agent."""

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import os

import config
from utils import plot_evaluation_results, save_results, print_statistics, create_directories


def evaluate_agent(model_path, n_episodes=100, render=False):
    """Evaluate a trained agent."""
    print("\n" + "="*70)
    print("EVALUATING TRAINED AGENT")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("="*70 + "\n")
    
    print("Loading model...")
    model = DQN.load(model_path)
    print("Model loaded successfully!\n")
    
    render_mode = 'human' if render else None
    env = gym.make(config.ENV_NAME, render_mode=render_mode)
    
    episode_rewards = []
    episode_lengths = []
    successful_landings = 0
    
    print("Starting evaluation...")
    print("-" * 70)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode_reward >= 200:
            successful_landings += 1
        
        if (episode + 1) % 10 == 0:
            recent_mean = np.mean(episode_rewards[-10:])
            print(f"Episodes {episode - 8}-{episode + 1}: Mean Reward = {recent_mean:.2f}")
    
    env.close()
    
    print("-" * 70)
    print("Evaluation completed!\n")
    
    results = {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "median_reward": float(np.median(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "successful_landings": successful_landings,
        "success_rate": successful_landings / n_episodes * 100,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }
    
    return results


def main():
    """Main evaluation function."""
    create_directories()
    
    model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model")
    
    if not os.path.exists(model_path + ".zip"):
        print(f"\nError: Model not found at {model_path}.zip")
        print("Please train the model first by running: python train.py")
        return
    
    results = evaluate_agent(model_path=model_path, n_episodes=100, render=False)
    
    print_statistics(results["episode_rewards"], results["episode_lengths"])
    
    print("Generating plots...")
    plot_evaluation_results(results["episode_rewards"], save_path='plots/evaluation_results.png')
    
    save_results(results, 'results/evaluation_results.json')
    
    print("="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Success Rate: {results['success_rate']:.1f}% ({results['successful_landings']}/{results['n_episodes']})")
    print(f"Best Episode: {results['max_reward']:.2f}")
    print(f"Worst Episode: {results['min_reward']:.2f}")
    print("="*70 + "\n")
    
    if results['mean_reward'] >= 200:
        print("Result: EXCELLENT - The agent has mastered lunar landing!")
    elif results['mean_reward'] >= 100:
        print("Result: GOOD - The agent performs well but can be improved.")
    elif results['mean_reward'] >= 0:
        print("Result: FAIR - The agent is learning but needs more training.")
    else:
        print("Result: POOR - The agent needs significantly more training.")
    
    print("\nResults saved to:")
    print("  - plots/evaluation_results.png")
    print("  - results/evaluation_results.json\n")


if __name__ == "__main__":
    main()
