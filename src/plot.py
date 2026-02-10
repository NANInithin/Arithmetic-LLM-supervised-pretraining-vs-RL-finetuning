import matplotlib.pyplot as plt
import numpy as np
import os
import mlflow #

def plot_graphs():
    # 1. Setup Paths
    supervised_path = os.path.join("logs", "supervised_loss.npy")
    rl_path = os.path.join("logs", "rl_rewards_replay.npy")

    # 2. Check if files exist
    if not os.path.exists(supervised_path) or not os.path.exists(rl_path):
        print(f"Error: Log files not found. Check if they are in 'logs/'.")
        return

    # 3. Load Data
    loss_data = np.load(supervised_path)
    reward_data = np.load(rl_path)

    # 4. Create Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Graph 1: Pretraining Loss ---
    epochs = range(1, len(loss_data) + 1)
    ax1.plot(epochs, loss_data, 'b-', linewidth=2, marker='o')
    ax1.set_title("Supervised Pretraining: Loss Curve (Reverse Mode)")
    ax1.grid(True, alpha=0.3)
    
    # --- Graph 2: RL Rewards ---
    episodes = range(1, len(reward_data) + 1)
    ax2.plot(episodes, reward_data, 'g-', label='Moving Avg Reward')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Accuracy')
    ax2.set_title("RL Fine-Tuning: Digit-Wise Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # 5. Save Locally
    output_path = "outputs/training_results.png"
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path)
    print(f"Graph saved locally as '{output_path}'")

    # 6. MLflow Artifact Logging
    # This logs the actual image file to the active MLflow run
    if mlflow.active_run():
        mlflow.log_artifact(output_path) #
        print("Successfully logged plot to MLflow artifacts.")
    else:
        print("No active MLflow run found to log artifact.")

    plt.show()

if __name__ == "__main__":
    plot_graphs()