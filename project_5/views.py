from django.shortcuts import render
from django.contrib import messages
import torch
import numpy as np
import os
import random

from pbl import settings
# Import your existing modules
from project_5.reinforcement_learning.policy_network import create_policy_network
from project_5.reinforcement_learning.trajectory_simulation import RL
from project_5.reinforcement_learning.rlhf import (
    sample_trajectories,
    build_pairwise_preferences_sim,
    RewardNet,
    train_reward_bt
)
from project_5.reinforcement_learning.retraining_with_RLHF import (
    evaluate_policy_organic_avoidance,
    reinforce_with_learned_reward
)
from project_5.reinforcement_learning.mouse import (
    initialize_grid_with_cheese_types,
    print_grid_with_cheese_types
)

from project_5.utils.visualization_utils import (
    grid_to_html,
    trajectory_to_html,
    compare_trajectories_html
)

# Store policy state between requests
_current_policy = None
_current_reward_net = None

MAX_PREFERENCES = 10
SAVE_MODEL_PATH = os.path.join(settings.BASE_DIR, 'data', 'project_5')


def index(request):
    global _current_policy, _current_reward_net

    # Initialize context with current states
    context = {
        'policy': _current_policy is not None,
        'reward_net': _current_reward_net is not None,
        'training_results': [],
        'sample_grid': None,  # For showing a sample grid
        'phase': 'initial',  # 'initial', 'task1', 'task2', 'task3'
    }

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'show_grid':
            # Show a sample grid visualization
            grid, mouse_pos, cheese_pos, organic_cheese_positions = initialize_grid_with_cheese_types()
            context['sample_grid'] = grid_to_html(grid, title="Sample Environment")
            messages.info(request, "Sample grid generated!")
        elif action == 'reset':
            # Reset everything
            _current_policy = None
            _current_reward_net = None

            # Remove saved files
            for file in ['current_policy.pt', 'reward_net_bt.pt', 'policy_task3_rlhf.pt', 'initial_policy.pt']:
                full_path = os.path.join(SAVE_MODEL_PATH, file)

                if os.path.exists(full_path):
                    try:
                        os.remove(full_path)
                        print(f"Removed {full_path}")
                    except:
                        pass

            messages.success(request, "Session reset successfully! All models cleared.")
            context['policy'] = False
            context['reward_net'] = False

        elif action == 'trajectory':
            # Create and store the policy
            _current_policy = create_policy_network()
            trajectory_simulation(request, context, _current_policy)
            context['policy'] = True

            # Save for persistence
            torch.save(_current_policy.state_dict(), os.path.join(SAVE_MODEL_PATH, "current_policy.pt"))
            messages.success(request, "Initial policy trained successfully!")

        elif action == 'train_sim':
            if _current_policy is not None:
                _current_reward_net = train_with_rlhf_sim(request, context, _current_policy)
                context['reward_net'] = True
                messages.success(request, "Reward model trained with simulated preferences!")
            else:
                messages.error(request, "Please train the initial policy first!")
        elif action == 'retrain':
            if _current_policy is not None and _current_reward_net is not None:
                retrain_policy(request, context, _current_policy, _current_reward_net)
                messages.success(request, "Policy retrained with RLHF!")
            else:
                messages.error(request, "Please complete Tasks 1 and 2 first!")

    return render(request, 'project5_base.html', context)


def sample_trajectory_check(request, context, policy):
    """ Generate a sample trajectory for visualization """

    if context['phase'] == 'task1':
        sample_traj = sample_trajectories(policy, K=1, time_horizon=20)[0]
        traj_html = trajectory_to_html(sample_traj, max_steps=5, title="Sample Trajectory After Training")

        context['training_results'].append({
            'task_name': 'Task 1: Initial Policy Training',
            'details': {
                'type': 'task1_complete',
                'trajectory_html': traj_html,
                'message': 'Training completed successfully! The policy network has learned basic navigation and cheese collection through REINFORCE algorithm.'
            }
        })
    elif context['phase'] == 'task3':
        sample_traj = sample_trajectories(policy, K=1, time_horizon=20)[0]
        traj_html = trajectory_to_html(sample_traj, max_steps=5, title="Sample Trajectory After Reward Model Training")

        context['training_results'].append({
            'task_name': 'Task 2: Reward Model Training',
            'details': {
                'type': 'task2_complete',
                'trajectory_html': traj_html,
                'message': 'Policy evaluated after reward model training'
            }
        })
    return None


def trajectory_simulation(request, context, policy):
    """Task 1: Train initial policy - modified to accept policy as parameter"""
    context['phase'] = 'task1'

    RL(policy, N_trajectories=1000)
    # Generate a sample trajectory for visualization
    sample_trajectory_check(request, context, policy)

    return None


def train_with_rlhf_sim(request, context, policy):
    """Task 2a: Train reward model - modified to accept and return reward_net"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    context['phase'] = 'task2'

    # Sample trajectories using the provided policy
    trajectories = sample_trajectories(policy, K=100, time_horizon=50)

    # Build preferences
    pairs = build_pairwise_preferences_sim(trajectories, provider="sim", max_pairs=120)

    # Train reward model
    reward_net = RewardNet()
    reward_net = train_reward_bt(
        reward_net, trajectories, pairs,
        epochs=20,  # More epochs
        lr=2e-3,  # Higher learning rate
        batch_size=6,  # Slightly smaller batch
        device=device
    )

    # Save the reward model
    torch.save(reward_net.state_dict(), os.path.join(SAVE_MODEL_PATH, "reward_net_bt.pt"))
    print("Saved reward_net_bt.pt")

    # Add results to context - pass data not HTML
    organic_hits = sum(t["organic_hits"] for t in trajectories)
    cheese_hits = sum(t["cheese_hits"] for t in trajectories)

    context['training_results'].append({
        'task_name': 'Task 2a: Reward Model Training',
        'details': {
            'type': 'task2a_complete',
            'trajectories_count': len(trajectories),
            'pairs_count': len(pairs),
            'organic_hits': organic_hits,
            'cheese_hits': cheese_hits,
            'epochs': 20,
            'device': device.upper(),
            'message': 'Reward model trained successfully using simulated preferences!'
        }
    })

    return reward_net


def retrain_policy(request, context, policy, reward_net):
    """Task 3: Retrain policy - modified to accept policy and reward_net"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluate before training
    print("Evaluation BEFORE Task3 training:")
    before_stats = evaluate_policy_organic_avoidance(policy, K=200)
    print(before_stats)

    # Retrain policy with learned reward + KL
    beta = 0.01  # Reduce from 0.003 to allow more learning

    trained_policy = reinforce_with_learned_reward(
        policy,
        reward_net,
        N_epochs=80,  # Reduce epochs
        trajectories_per_epoch=30,  # Fewer trajectories per epoch
        time_horizon=50,
        gamma=0.99,
        beta=beta,
        lr=2e-3,  # Higher learning rate
        device=device,
        normalize_returns=True,
        verbose_every=5  # More frequent updates
    )

    # Evaluate after training
    print("Evaluation AFTER Task3 training:")
    after_stats = evaluate_policy_organic_avoidance(trained_policy, K=200)
    print(after_stats)

    # Save the trained policy
    torch.save(trained_policy.state_dict(), os.path.join(SAVE_MODEL_PATH, "policy_task3_rlhf.pt"))
    print("Saved trained policy to policy_task3_rlhf.pt")

    # Update global policy
    global _current_policy
    _current_policy = trained_policy

    # Add results to context - pass data not HTML
    improvement = before_stats["total_organic_hits"] - after_stats["total_organic_hits"]
    improvement_pct = (improvement / before_stats["total_organic_hits"] * 100) if before_stats[
                                                                                      "total_organic_hits"] > 0 else 0

    context['training_results'].append({
        'task_name': 'Task 3: RLHF Policy Retraining',
        'details': {
            'type': 'task3_complete',
            'before_stats': before_stats,
            'after_stats': after_stats,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'beta': beta,
            'epochs': 80,
            'lr': 2e-3,
            'device': device.upper(),
            'message': 'Policy successfully retrained with learned human preferences!'
        }
    })

    return None


# Load saved models on startup if they exist
def load_saved_states():
    global _current_policy, _current_reward_net

    policy_path = os.path.join(SAVE_MODEL_PATH, "current_policy.pt")
    reward_path = os.path.join(SAVE_MODEL_PATH, "reward_net_bt.pt")

    if os.path.exists(policy_path):
        try:
            _current_policy = create_policy_network()
            _current_policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
            print("Loaded saved policy")
        except Exception as e:
            print(f"Failed to load policy: {e}")
            pass

    if os.path.exists(reward_path):
        try:
            _current_reward_net = RewardNet()
            _current_reward_net.load_state_dict(torch.load(reward_path, map_location="cpu"))
            print("Loaded saved reward net")
        except Exception as e:
            print(f"Failed to load reward net: {e}")
            pass


# Initialize on import
load_saved_states()
