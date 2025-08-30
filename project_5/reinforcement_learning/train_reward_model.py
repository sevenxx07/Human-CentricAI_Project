# train_reward_model.py (example driver)

import torch
from project_5.reinforcement_learning.policy_network import create_policy_network
from project_5.reinforcement_learning.rlhf import sample_trajectories, build_pairwise_preferences_sim, RewardNet, train_reward_bt

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load or create your current policy
    policy = create_policy_network()
    policy.eval()

    # 2) Sample MORE trajectories for better diversity
    trajectories = sample_trajectories(policy, K=100, time_horizon=50)  # Increased from 80

    # Print some stats to see what we got
    organic_hits = sum(t["organic_hits"] for t in trajectories)
    cheese_hits = sum(t["cheese_hits"] for t in trajectories)
    print(f"Collected {len(trajectories)} trajectories:")
    print(f"  Total organic cheese hits: {organic_hits}")
    print(f"  Total normal cheese hits: {cheese_hits}")

    # 3) Build pairwise preferences
    pairs = build_pairwise_preferences_sim(trajectories, provider="sim", max_pairs=120)

    # 4) Train the reward model with BETTER parameters
    reward_net = RewardNet()
    reward_net = train_reward_bt(
        reward_net, trajectories, pairs,
        epochs=20,  # More epochs
        lr=2e-3,  # Higher learning rate
        batch_size=6,  # Slightly smaller batch
        device=device
    )

    # 5) Save reward model
    torch.save(reward_net.state_dict(), "reward_net_bt.pt")
    print("Saved reward_net_bt.pt")
