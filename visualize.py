"""Visualization script to watch trained agent perform lunar landings."""

import gymnasium as gym
from stable_baselines3 import DQN
import os
import time

import config


def visualize_agent(model_path, n_episodes=5, delay=0.01):
    """Visualize a trained agent performing lunar landings."""
    print("\n" + "="*70)
    print("LUNAR LANDER VISUALIZATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("="*70 + "\n")
    
    print("Loading model...")
    model = DQN.load(model_path)
    print("Model loaded successfully!\n")
    
    env = gym.make(config.ENV_NAME, render_mode='human')
    
    print("Starting visualization...")
    print("Close the window to stop.\n")
    
    for episode in range(n_episodes):
        print(f"\n{'='*70}")
        print(f"EPISODE {episode + 1}/{n_episodes}")
        print('='*70)
        
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
            time.sleep(delay)
        
        print(f"\nEpisode {episode + 1} Results:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length} steps")
        
        if episode_reward >= 200:
            print("  Status: SUCCESSFUL LANDING")
        elif episode_reward >= 0:
            print("  Status: Landed (suboptimal)")
        else:
            print("  Status: Crashed")
        
        if episode < n_episodes - 1:
            print("\nNext episode in 2 seconds...")
            time.sleep(2)
    
    env.close()
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETED")
    print("="*70 + "\n")


def main():
    """Main visualization function."""
    model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model")
    
    if not os.path.exists(model_path + ".zip"):
        print(f"\nError: Model not found at {model_path}.zip")
        print("Please train the model first by running: python train.py")
        
        final_model_path = os.path.join(config.MODEL_SAVE_PATH, "final_model")
        if os.path.exists(final_model_path + ".zip"):
            print(f"\nUsing final model: {final_model_path}")
            model_path = final_model_path
        else:
            return
    
    print("\n" + "="*70)
    print("LUNAR LANDER ACTIONS:")
    print("="*70)
    print("  0: Do nothing")
    print("  1: Fire left engine")
    print("  2: Fire main engine")
    print("  3: Fire right engine")
    print("="*70)
    
    visualize_agent(model_path=model_path, n_episodes=5, delay=0.01)


if __name__ == "__main__":
    main()
