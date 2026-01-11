"""Training script for Lunar Lander DQN Agent."""

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch
import os

import config
from utils import create_directories, save_results


def create_environment(render_mode=None):
    """Create and wrap the Lunar Lander environment."""
    env = gym.make(config.ENV_NAME, render_mode=render_mode)
    env = Monitor(env)
    return env


def train_agent(total_timesteps=None, verbose=1):
    """Train the DQN agent on Lunar Lander environment."""
    if total_timesteps is None:
        total_timesteps = config.TOTAL_TIMESTEPS
    
    print("\n" + "="*70)
    print("TRAINING LUNAR LANDER DQN AGENT")
    print("="*70)
    print(f"Environment: {config.ENV_NAME}")
    print(f"Algorithm: Deep Q-Network (DQN)")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Gamma: {config.GAMMA}")
    print("="*70 + "\n")
    
    create_directories()
    
    env = create_environment()
    eval_env = create_environment()
    
    print("Initializing DQN model...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=config.LEARNING_RATE,
        buffer_size=config.BUFFER_SIZE,
        learning_starts=config.LEARNING_STARTS,
        batch_size=config.BATCH_SIZE,
        tau=config.TAU,
        gamma=config.GAMMA,
        train_freq=config.TRAIN_FREQ,
        gradient_steps=config.GRADIENT_STEPS,
        target_update_interval=config.TARGET_UPDATE_INTERVAL,
        exploration_fraction=config.EXPLORATION_FRACTION,
        exploration_initial_eps=config.EXPLORATION_INITIAL_EPS,
        exploration_final_eps=config.EXPLORATION_FINAL_EPS,
        policy_kwargs=dict(net_arch=config.NET_ARCH),
        tensorboard_log=config.TENSORBOARD_LOG,
        verbose=verbose,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Device: {model.device}")
    print(f"Network Architecture: {config.NET_ARCH}\n")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.MODEL_SAVE_PATH,
        log_path=config.RESULTS_DIR,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ,
        save_path=config.MODEL_SAVE_PATH,
        name_prefix="checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    print("Starting training...")
    print("-" * 70)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=False,
            log_interval=100
        )
        print("\n" + "-" * 70)
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    final_model_path = os.path.join(config.MODEL_SAVE_PATH, "final_model")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    training_config = {
        "environment": config.ENV_NAME,
        "algorithm": "DQN",
        "total_timesteps": total_timesteps,
        "learning_rate": config.LEARNING_RATE,
        "buffer_size": config.BUFFER_SIZE,
        "batch_size": config.BATCH_SIZE,
        "gamma": config.GAMMA,
        "net_arch": config.NET_ARCH,
        "device": str(model.device)
    }
    save_results(training_config, "results/training_config.json")
    
    env.close()
    eval_env.close()
    
    return model


def quick_test(model, n_episodes=5):
    """Test the trained model."""
    print("\n" + "="*70)
    print(f"TESTING - Running {n_episodes} episodes")
    print("="*70)
    
    env = create_environment()
    episode_rewards = []
    episode_lengths = []
    
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
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    env.close()
    
    print(f"\nAverage Reward: {np.mean(episode_rewards):.2f} (+/- {np.std(episode_rewards):.2f})")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    model = train_agent(verbose=1)
    quick_test(model, n_episodes=5)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print("\nNext steps:")
    print("1. Run 'python evaluate.py' to evaluate the trained agent")
    print("2. Run 'python visualize.py' to watch the agent in action")
    print("3. Run 'tensorboard --logdir=./tensorboard_logs/' to view metrics")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
