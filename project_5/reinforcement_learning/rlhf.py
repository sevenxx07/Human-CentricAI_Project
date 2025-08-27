import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Reuse your existing imports & constants
from mouse import (
    initialize_grid_with_cheese_types,
    print_grid_with_cheese_types,
    move,
    get_reward,
    ACTIONS,
    MOUSE,
    GRID_SIZE,
    ACTION_TO_DELTA,
    CHEESE,
    ORGANIC_CHEESE,
    TRAP
)
from policy_network import create_policy_network


# ---------- Helpers reused from your code ----------
def state_to_tensor(grid):
    """One-hot CxHxW tensor with C=6 channels, H=W=5."""
    one_hot = np.eye(6)[grid]
    return torch.tensor(one_hot, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


def select_action(policy, grid):
    """Sample an action from the policy. Your policy already ends with Softmax."""
    state = state_to_tensor(grid)
    with torch.no_grad():
        probs = policy(state)  # shape: (1, 4), already probabilities
    dist = torch.distributions.Categorical(probs.squeeze(0))
    a_idx = dist.sample()
    return ACTIONS[a_idx.item()], torch.log(probs.squeeze(0)[a_idx])


# ---------- Trajectory collection ----------
def collect_trajectory(policy, time_horizon=50, gamma=0.99, stop_on_terminal=True):
    """
    Rollout one trajectory with the current policy.
    Returns:
        states:  list of np.array (grid snapshots)
        actions: list of str actions
        rewards: list of float env rewards (your original rewards; used only for logging)
        info:    dict with counters (cheese hits, organic hits, terminal flag)
    """
    grid, mouse_pos, cheese_pos, organic_cheese_positions = initialize_grid_with_cheese_types()

    states, actions, rewards = [], [], []
    cheese_hits, organic_hits, terminal = 0, 0, False

    for t in range(time_horizon):
        states.append(grid.copy())
        action, _ = select_action(policy, grid)
        grid, cell_content = move(action, grid)
        actions.append(action)
        r = get_reward(cell_content)
        rewards.append(r)

        if cell_content == CHEESE:
            cheese_hits += 1
            if stop_on_terminal:
                terminal = True
        elif cell_content == ORGANIC_CHEESE:
            organic_hits += 1
            if stop_on_terminal:
                terminal = True

        # trap also terminal in your env
        if r == -50:  # trap
            if stop_on_terminal:
                terminal = True

        if terminal:
            break

    return {
        "states": states,  # each is 5x5 int grid
        "actions": actions,  # strings
        "rewards": rewards,  # floats
        "cheese_hits": cheese_hits,
        "organic_hits": organic_hits,
        "terminal": terminal,
    }


def sample_trajectories(policy, K=20, time_horizon=50):
    """Collect K trajectories."""
    trajs = []
    for _ in range(K):
        trajs.append(collect_trajectory(policy, time_horizon=time_horizon))
    return trajs


# ---------- Preference providers ----------
def show_trajectory_text(traj, title="Trajectory"):
    """Pretty-print a trajectory for a human to judge."""
    print(
        f"\n=== {title} | len={len(traj['states'])} | cheese={traj['cheese_hits']} | organic={traj['organic_hits']} ===")
    for step, (grid, act, rew) in enumerate(zip(traj['states'], traj['actions'], traj['rewards'])):
        print(f"\nStep {step}: action={act}, env_reward={rew:.2f}")
        print_grid_with_cheese_types(grid)


def human_preference(traj_a, traj_b):
    """
    Ask a human to pick the better trajectory.
    Returns 1 if A preferred, 0 if B preferred.
    """
    show_trajectory_text(traj_a, "Trajectory A")
    show_trajectory_text(traj_b, "Trajectory B")
    while True:
        pick = input("\nWhich trajectory do you prefer? [A/B]: ").strip().lower()
        if pick in ("a", "b"):
            return 1 if pick == "a" else 0
        print("Please answer A or B.")


def simulated_preference(traj_a, traj_b):
    """
    FIXED: Simulated feedback rule that prefers AVOIDING organic cheese.
    - Primary: Prefer the one with FEWER ORGANIC cheese hits (avoid organic)
    - Secondary: If organic hits are equal, prefer MORE normal cheese hits (get food)
    - Tertiary: If both equal, prefer shorter trajectory (efficiency)
    Returns 1 if A preferred, 0 if B preferred.
    """
    oa, ob = traj_a["organic_hits"], traj_b["organic_hits"]
    ca, cb = traj_a["cheese_hits"], traj_b["cheese_hits"]

    # Primary criterion: FEWER organic hits is better
    if oa != ob:
        return 1 if oa < ob else 0  # A preferred if A has fewer organic hits

    # Secondary: MORE normal cheese hits is better (when organic hits are equal)
    if ca != cb:
        return 1 if ca > cb else 0  # A preferred if A has more normal cheese

    # Tertiary: shorter trajectory is better (efficiency)
    la, lb = len(traj_a["states"]), len(traj_b["states"])
    if la != lb:
        return 1 if la < lb else 0  # A preferred if A is shorter

    # Random tie-breaker
    return np.random.randint(0, 2)


def build_pairwise_preferences_sim(trajectories, provider="sim", max_pairs=100):
    """
    Create pairwise (i, j, y) with y=1 if traj i preferred over j, else 0.
    provider: "sim" or "human"
    """
    prefs = []
    N = len(trajectories)
    choose = simulated_preference if provider == "sim" else human_preference

    # Create more diverse pairs, not just consecutive ones
    pairs_created = 0
    for i in range(N):
        for j in range(i + 1, N):
            if pairs_created >= max_pairs:
                break

            # Prioritize pairs where organic hits differ significantly
            oa, ob = trajectories[i]["organic_hits"], trajectories[j]["organic_hits"]
            if abs(oa - ob) > 0 or np.random.random() < 0.3:  # Include some random pairs too
                y = choose(trajectories[i], trajectories[j])
                prefs.append((i, j, y))
                pairs_created += 1

        if pairs_created >= max_pairs:
            break

    return prefs

class RewardNet(nn.Module):
    """
    CNN reward model: input Cx5x5 (C=6 one-hot channels), output scalar r(s).
    Trajectory score S = sum_t r(s_t).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Added depth
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),  # Increased capacity
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # scalar reward per state
        )

    def forward(self, state_tensor_bchw):
        # state_tensor_bchw: (B, 6, 5, 5)
        return self.net(state_tensor_bchw).squeeze(-1)  # (B,)


def score_trajectory(reward_net, traj, device="cpu"):
    """
    Sum predicted rewards over states in a trajectory.
    """
    states = traj["states"]
    with torch.no_grad():
        batch = torch.cat([state_to_tensor(s) for s in states], dim=0)  # (T, 6, 5, 5)
        batch = batch.to(device)
        per_state_rewards = reward_net(batch)  # (T,)
        return per_state_rewards.sum().item()


class PairwisePrefDataset(Dataset):
    """
    Holds (trajectory_i, trajectory_j, label y∈{0,1}) and feeds state tensors to reward model.
    """

    def __init__(self, trajectories, pairs):
        self.trajs = trajectories
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, y = self.pairs[idx]
        ti, tj = self.trajs[i], self.trajs[j]
        return ti["states"], tj["states"], torch.tensor([y], dtype=torch.float32)


def collate_pairs(batch):
    """
    Collate variable-length trajectories.
    """
    return batch


def train_reward_bt(
        reward_net, trajectories, pairs,
        epochs=20, lr=1e-3, batch_size=4, device="cpu"
):
    """
    IMPROVED: Train reward_net with Bradley–Terry loss on pairwise preferences.
    """
    print(f"Training reward model with {len(pairs)} preference pairs...")

    reward_net.to(device)
    dataset = PairwisePrefDataset(trajectories, pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pairs, num_workers=0)

    opt = optim.Adam(reward_net.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        total_loss, n = 0.0, 0
        correct = 0
        total = 0

        for batch in loader:
            opt.zero_grad()
            logits_list, labels_list = [], []

            for states_i, states_j, y in batch:
                # Build tensors for each trajectory and sum predicted rewards
                batch_i = torch.cat([state_to_tensor(s) for s in states_i], dim=0).to(device)
                batch_j = torch.cat([state_to_tensor(s) for s in states_j], dim=0).to(device)

                Si = reward_net(batch_i).sum()
                Sj = reward_net(batch_j).sum()

                # Bradley–Terry logit is (Si - Sj)
                logit = (Si - Sj).unsqueeze(0)  # shape (1,)
                logits_list.append(logit)
                labels_list.append(y.to(device))

            if logits_list:  # Check if we have any data
                logits = torch.cat(logits_list, dim=0)
                labels = torch.cat(labels_list, dim=0)

                loss = criterion(logits, labels)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(reward_net.parameters(), max_norm=1.0)

                opt.step()

                total_loss += loss.item() * logits.size(0)
                n += logits.size(0)

                # Accuracy for sanity check
                with torch.no_grad():
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.numel()

        if n > 0:
            avg_loss = total_loss / n
            accuracy = correct / total if total > 0 else 0
            print(f"[RewardNet][Epoch {ep}] loss={avg_loss:.4f} pref-acc={accuracy:.3f}")

        # Early stopping based on accuracy
        if accuracy > 0.95 and ep > 10:
            print("Early stopping: High accuracy achieved")
            break

    return reward_net