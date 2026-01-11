"""Quick start menu for Lunar Lander DQN Agent."""

import os


def print_menu():
    """Print the main menu."""
    print("\n" + "="*70)
    print("LUNAR LANDER DQN AGENT")
    print("="*70)
    print("\nOptions:\n")
    print("1. Train agent (full - 500k timesteps)")
    print("2. Evaluate trained agent")
    print("3. Visualize agent")
    print("4. Quick demo (50k timesteps)")
    print("5. Launch TensorBoard")
    print("6. Exit")
    print("\n" + "="*70)


def train_full():
    """Train the full model."""
    print("\nStarting full training (500,000 timesteps)...")
    print("Estimated time: 2-4 hours on CPU.\n")
    import train
    train.main()


def train_quick():
    """Quick training demo."""
    print("\nStarting quick demo (50,000 timesteps)...")
    print("Estimated time: 10-15 minutes.\n")
    import train
    model = train.train_agent(total_timesteps=50000, verbose=1)
    train.quick_test(model, n_episodes=3)


def evaluate():
    """Evaluate the agent."""
    if not os.path.exists("models/lunar_lander_dqn/best_model.zip") and \
       not os.path.exists("models/lunar_lander_dqn/final_model.zip"):
        print("\nNo trained model found.")
        print("Please train the agent first (Option 1 or 4).\n")
        return
    
    print("\nStarting evaluation...")
    import evaluate as eval_module
    eval_module.main()


def visualize():
    """Visualize the agent."""
    if not os.path.exists("models/lunar_lander_dqn/best_model.zip") and \
       not os.path.exists("models/lunar_lander_dqn/final_model.zip"):
        print("\nNo trained model found.")
        print("Please train the agent first (Option 1 or 4).\n")
        return
    
    print("\nStarting visualization...")
    import visualize as vis_module
    vis_module.main()


def tensorboard():
    """Launch TensorBoard."""
    print("\nLaunching TensorBoard...")
    print("Open http://localhost:6006 in your browser.")
    print("Press Ctrl+C to stop.\n")
    os.system("tensorboard --logdir=./tensorboard_logs/")


def main():
    """Main menu loop."""
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == "1":
                train_full()
            elif choice == "2":
                evaluate()
            elif choice == "3":
                visualize()
            elif choice == "4":
                train_quick()
            elif choice == "5":
                tensorboard()
            elif choice == "6":
                print("\nGoodbye!\n")
                break
            else:
                print("\nInvalid choice. Please enter a number between 1 and 6.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    print("\nWelcome to the Lunar Lander DQN Agent!")
    main()
