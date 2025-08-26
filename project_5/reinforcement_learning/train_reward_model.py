# train_reward_model.py (example driver)

import torch
from policy_network import create_policy_network
from rlhf import sample_trajectories, build_pairwise_preferences, RewardNet, train_reward_bt

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load or create your current policy (already trained with Task 1, or untrained to start)
    policy = create_policy_network()
    policy.eval()   # we don't update policy in Task 2

    # 2) Sample trajectories from current policy
    trajectories = sample_trajectories(policy, K=40, time_horizon=50)

    # 3) Build pairwise preferences (use "human" to prompt in terminal)
    pairs = build_pairwise_preferences(trajectories, provider="sim")  # "sim" or "human"

    # 4) Train the reward model with BT loss
    reward_net = RewardNet()
    reward_net = train_reward_bt(
        reward_net, trajectories, pairs,
        epochs=10, lr=1e-3, batch_size=8, device=device
    )

    # 5) (Optional) Save reward model for Task 3
    torch.save(reward_net.state_dict(), "reward_net_bt.pt")
