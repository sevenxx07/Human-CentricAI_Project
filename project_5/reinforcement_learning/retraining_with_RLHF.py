# task3_rlhf.py
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from policy_network import create_policy_network
from rlhf import (
    state_to_tensor,
    sample_trajectories,
    collect_trajectory,
    RewardNet,
    score_trajectory,
    ACTIONS,
    initialize_grid_with_cheese_types,
    move,
    GRID_SIZE,
    CHEESE, ORGANIC_CHEESE, TRAP
)

# ---------- Utilities ----------

def probs_from_policy(policy, grid, device="cpu"):
    """Return action probability vector (1x4) for given grid."""
    state = state_to_tensor(grid).to(device)  # (1,6,5,5)
    with torch.no_grad():
        p = policy(state).squeeze(0)  # (4,)
    return p

def sample_action_and_logprob(policy, grid, device="cpu"):
    """Sample action under policy, return action index, action string, log_prob, and action probs."""
    state = state_to_tensor(grid).to(device)
    probs = policy(state).squeeze(0)  # shape (4,)
    dist = Categorical(probs)
    a_idx = dist.sample()
    log_prob = dist.log_prob(a_idx)
    return a_idx.item(), ACTIONS[a_idx.item()], log_prob, probs

# ---------- REINFORCE with learned reward + KL penalty ----------

def reinforce_with_learned_reward(
    policy,
    reward_net,
    N_epochs=200,
    trajectories_per_epoch=40,
    time_horizon=50,
    gamma=0.99,
    beta=0.01,
    lr=1e-3,
    device="cpu",
    normalize_returns=True,
    verbose_every=10
):
    """
    Retrain `policy` with REINFORCE using learned reward_net(s).
    Adds KL penalty between current policy and a frozen copy of the policy before the epoch update.
    """

    policy.to(device)
    reward_net.to(device)
    reward_net.eval()  # reward is fixed

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(1, N_epochs + 1):
        # freeze old policy (copy) for KL computations
        old_policy = copy.deepcopy(policy).to(device)
        old_policy.eval()
        # collect trajectories with current policy (on-policy)
        batch_trajs = []
        for k in range(trajectories_per_epoch):
            traj = collect_trajectory(policy, time_horizon=time_horizon)
            batch_trajs.append(traj)

        # Build loss across the batch
        optimizer.zero_grad()
        total_loss = 0.0
        total_steps = 0

        # We'll sum losses from each trajectory then divide by number of trajectories
        for traj in batch_trajs:
            states = traj["states"]
            actions = traj["actions"]
            # We'll need per-step log_probs and per-step learned rewards
            log_probs = []
            learned_rewards = []

            # Roll the trajectory again to collect log_probs given the current policy
            for s_grid, a_str in zip(states, actions):
                state = state_to_tensor(s_grid).to(device)  # (1,6,5,5)
                probs = policy(state).squeeze(0)            # (4,) - requires grad
                dist = Categorical(probs)
                a_idx = ACTIONS.index(a_str)
                log_p = dist.log_prob(torch.tensor(a_idx, device=device))
                log_probs.append(log_p)
            grid_reset, _, _, _ = initialize_grid_with_cheese_types()
            for s_grid in states:
                st_tensor = state_to_tensor(s_grid).to(device)
                # reward_net returns scalar per state
                r_hat = reward_net(st_tensor).detach().cpu().item()
                learned_rewards.append(r_hat)

            # Now compute discounted returns on learned_rewards (per-trajectory)
            returns = []
            G = 0.0
            for r in reversed(learned_rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)

            if normalize_returns and len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            elif normalize_returns:
                returns = returns - returns.mean()

            # Policy loss (REINFORCE)
            traj_policy_loss = 0.0
            # KL penalty for this trajectory (sum over states)
            traj_kl = 0.0

            for log_p, Gt, s_grid in zip(log_probs, returns, states):
                traj_policy_loss = traj_policy_loss - log_p * Gt  # negative because PyTorch does gradient descent
                # KL between current policy and old policy at this state
                # Get probability vectors
                state = state_to_tensor(s_grid).to(device)
                p = policy(state).squeeze(0)       # (4,) - with grad
                with torch.no_grad():
                    p_old = old_policy(state).squeeze(0)  # (4,), no grad
                # compute D_KL(p || p_old) = sum p * (log p - log p_old)
                # Ensure numerical stability with clamp
                eps = 1e-8
                p_clamped = (p + eps)
                p_old_clamped = (p_old + eps)
                traj_kl = traj_kl + torch.sum(p_clamped * (torch.log(p_clamped) - torch.log(p_old_clamped)))

            # Combine
            traj_loss = traj_policy_loss + beta * traj_kl
            total_loss = total_loss + traj_loss
            total_steps += len(states)

        # Average loss across trajectories
        total_loss = total_loss / len(batch_trajs)
        total_loss.backward()
        optimizer.step()

        if epoch % verbose_every == 0 or epoch == 1:
            # Evaluate on a held-out sample to see organic cheese hits
            with torch.no_grad():
                eval_trajs = sample_trajectories(policy, K=50)
                organic_hits = sum(t["organic_hits"] for t in eval_trajs)
                cheese_hits = sum(t["cheese_hits"] for t in eval_trajs)
            print(f"[Epoch {epoch}] Loss={total_loss.item():.4f} | Eval organic_hits={organic_hits} | normal_cheese_hits={cheese_hits}")

    return policy

# ---------- Evaluation helpers ----------

def evaluate_policy_organic_avoidance(policy, K=200, time_horizon=50):
    """Return fraction of trajectories that hit organic cheese and average counts."""
    trajs = sample_trajectories(policy, K=K, time_horizon=time_horizon)
    total_org_hits = sum(t["organic_hits"] for t in trajs)
    total_cheese_hits = sum(t["cheese_hits"] for t in trajs)
    frac_with_organic = sum(1 for t in trajs if t["organic_hits"] > 0) / K
    return {
        "K": K,
        "total_organic_hits": total_org_hits,
        "total_normal_cheese_hits": total_cheese_hits,
        "frac_trajectories_with_organic": frac_with_organic,
        "avg_organic_per_traj": total_org_hits / K,
        "avg_normal_cheese_per_traj": total_cheese_hits / K
    }

# ---------- Example driver ----------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Create policy and load reward_net (trained in Task 2)
    policy = create_policy_network()
    reward_net = RewardNet()
    try:
        reward_net.load_state_dict(torch.load("reward_net_bt.pt", map_location=device))
        print("Loaded reward_net_bt.pt")
    except Exception as e:
        print("Could not load reward_net_bt.pt — make sure you trained Task2 and saved it. Error:", e)
        # still continue (reward_net randomly initialized) but results won't be meaningful

    # optional: evaluate before training
    print("Evaluation BEFORE Task3 training:")
    before_stats = evaluate_policy_organic_avoidance(policy, K=200)
    print(before_stats)

    # 2) Retrain policy with learned reward + KL
    beta = 0.003   # small KL weight; tune (e.g. 0.005 - 0.05)
    policy = policy.to(device)
    trained_policy = reinforce_with_learned_reward(
        policy,
        reward_net,
        N_epochs=120,
        trajectories_per_epoch=32,
        time_horizon=50,
        gamma=0.99,
        beta=beta,
        lr=1e-3,
        device=device,
        normalize_returns=True,
        verbose_every=10
    )

    # 3) Evaluate after training
    print("Evaluation AFTER Task3 training:")
    after_stats = evaluate_policy_organic_avoidance(trained_policy, K=200)
    print(after_stats)

    # Optionally save the trained policy
    torch.save(trained_policy.state_dict(), "policy_task3_rlhf.pt")
    print("Saved trained policy to policy_task3_rlhf.pt")
