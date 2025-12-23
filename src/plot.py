import matplotlib.pyplot as plt
import numpy as np
import os

def plot_graphs():
    # 1. Load Data
    if not os.path.exists("supervised_loss.npy") or not os.path.exists("rl_rewards_replay.npy"):
        print("Error: Log files not found. Please run training scripts first.")
        return

    loss_data = np.load("supervised_loss.npy")
    reward_data = np.load("rl_rewards_replay.npy")

    # 2. Setup Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Graph 1: Pretraining Loss ---
    epochs = range(1, len(loss_data) + 1)
    ax1.plot(epochs, loss_data, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title("Supervised Pretraining: Loss Curve ", fontsize=14)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, alpha=0.3)
    
    # Annotate the final loss
    final_loss = loss_data[-1]
    ax1.annotate(f'Final Loss: {final_loss:.4f}', 
                 xy=(len(loss_data), final_loss), 
                 xytext=(len(loss_data)-5, final_loss+0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # --- Graph 2: RL Rewards ---
    episodes = range(1, len(reward_data) + 1)
    ax2.plot(episodes, reward_data, 'g-', linewidth=1.5, label='Moving Avg Reward')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Accuracy')
    ax2.set_title("RL Fine-Tuning: Accuracy Improvement ", fontsize=14)
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Average Reward (Smoothed)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("Success! Graph saved as 'training_results.png'")
    plt.show()

if __name__ == "__main__":
    plot_graphs()