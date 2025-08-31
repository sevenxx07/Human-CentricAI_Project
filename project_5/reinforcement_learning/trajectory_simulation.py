import torch
import torch.optim as optim
import numpy as np

from project_5.reinforcement_learning.mouse import (
    initialize_grid_with_cheese_types,
    move,
    get_reward,
    ACTIONS)

from project_5.reinforcement_learning.policy_network import create_policy_network

# Converting the grid to a one-hot encoded tensor suitable for the netwok
def state_to_tensor(grid):
    one_hot = np.eye(6)[grid]
    return torch.tensor(one_hot, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


# Probabilitstic action selection given the policy and the grid
def select_action(policy, grid):
    state = state_to_tensor(grid)  # One-hot encoding of the grid
    logits = policy(state)
    probabilities = torch.softmax(logits, dim=1)  # Probabilites over actions given a state
    distribution = torch.distributions.Categorical(probabilities)  # Categorical distribution
    actions_index = distribution.sample()  # Sampling an action from the distribution
    return ACTIONS[actions_index.item()], distribution.log_prob(
        actions_index)  # Returning the selected action and the log probability of it (needed for the reinforce gradient)


# Reainforcement learning loop.
def RL(policy, N_trajectories=40, gamma=0.99, time_horizon=50, N_epochs=200, eval_every=100):
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    # Track training statistics
    epoch_rewards = []
    epoch_cheese_hits = []
    epoch_trap_hits = []
    total_episodes = 0

    for epoch in range(N_epochs):
        # Tracking the times the mouse hits cheese or a trap
        batch_loss = 0
        total_rewards = 0
        cheese_encounters = 0
        trap_encounters = 0
        successful_episodes = 0

        for trajectory in range(N_trajectories):
            grid, mouse_pos, cheese_pos, organic_cheese_positions = initialize_grid_with_cheese_types()  # Resetting the envionment at the start of each trajectory
            log_probabilites = []
            rewards = []
            episode_reward = 0

            for time_step in range(time_horizon):
                action, log_prob = select_action(policy, grid)  # Select action
                prev_mouse_pos = tuple(np.argwhere(grid == 1)[0])  # Position of mouse before playing an action (moving)
                grid, cell_content = move(action, grid)  # Move mouse --> Get the new grid and the cell content
                new_mouse_pos = tuple(np.argwhere(grid == 1)[0])
                reward = get_reward(cell_content)  # Get reward based on the cell content

                if reward == 10:
                    cheese_encounters += 1
                    successful_episodes += 1
                elif reward == -50:
                    trap_encounters += 1

                log_probabilites.append(log_prob)
                rewards.append(reward)
                episode_reward += reward

                if reward == 10 or reward == -50:
                    break  # Stop early if cheese or trap is encountered

            total_rewards += episode_reward
            total_episodes += 1

            # Computing returns for each step
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)

            # Normalizing returns to stabilize learning
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            else:
                returns = (returns - returns.mean())

                # Policy loss calculaed using the reinforce gradient
            loss = 0
            for log_prob, G in zip(log_probabilites, returns):
                loss += -log_prob * G  # Negative because reinforce is gradient ascent but PyTorch is descent
                batch_loss += loss

            # Gradient update
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Store epoch statistics
        avg_reward = total_rewards / N_trajectories
        epoch_rewards.append(avg_reward)
        epoch_cheese_hits.append(cheese_encounters)
        epoch_trap_hits.append(trap_encounters)

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{N_epochs}: "
                f"Avg reward = {avg_reward:.2f}, "
                f"Cheese hits = {cheese_encounters}, Trap hits = {trap_encounters}"
            )

    # Calculate final statistics
    final_stats = {
        'avg_reward': sum(epoch_rewards[-10:]) / min(10, len(epoch_rewards)),  # Average of last 10 epochs
        'cheese_encounters': sum(epoch_cheese_hits[-10:]) // min(10, len(epoch_cheese_hits)),
        # Average of last 10 epochs
        'trap_encounters': sum(epoch_trap_hits[-10:]) // min(10, len(epoch_trap_hits)),  # Average of last 10 epochs
        'success_rate': (sum(epoch_cheese_hits[-10:]) / (N_trajectories * min(10, len(epoch_cheese_hits)))) * 100,
        # Success rate in last 10 epochs
        'total_episodes': total_episodes
    }

    return final_stats


if __name__ == "__main__":
    policy = create_policy_network()
    stats = RL(policy)
    print("Final training statistics:", stats)