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
SAVE_MODEL_PATH = os.path.join(settings.BASE_DIR, 'data', 'project_5_models')


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

        elif action == 'train_human':
            if _current_policy is not None:
                # Show trajectory comparison interface
                trajectories = sample_trajectories(_current_policy, K=10, time_horizon=30)

                # Select two trajectories with different characteristics for comparison
                traj_pair = random.sample(trajectories, 2)

                # Get CSRF token
                from django.middleware.csrf import get_token
                csrf_token = get_token(request)

                # Generate comparison HTML with CSRF token
                comparison_html = compare_trajectories_html(traj_pair[0], traj_pair[1])

                # Store trajectories in session for when user makes choice
                request.session['traj_pair'] = [traj_pair[0], traj_pair[1]]
                request.session['comparison_mode'] = True

                context['show_comparison'] = True
                context['comparison_html'] = comparison_html
                messages.info(request, "Please select which trajectory you prefer")
            else:
                messages.error(request, "Please train the initial policy first!")

        elif action == 'submit_preference':
            # Handle the preference submission
            preference = request.POST.get('preference')  # 'A' or 'B'
            if preference and request.session.get('comparison_mode'):
                # Get stored trajectories
                traj_pair = request.session.get('traj_pair')

                # Initialize preferences list if not exists
                if 'preferences' not in request.session:
                    request.session['preferences'] = []

                # Record preference: 1 if A preferred, 0 if B
                pref_value = 1 if preference == 'A' else 0
                prefs_list = request.session['preferences']
                prefs_list.append({
                    'traj_a': traj_pair[0],
                    'traj_b': traj_pair[1],
                    'preference': pref_value
                })
                request.session['preferences'] = prefs_list
                request.session.modified = True  # Ensure session is saved

                # Check if we have enough preferences
                num_prefs = len(request.session.get('preferences', []))
                if num_prefs < MAX_PREFERENCES:  # Collect 10 preferences
                    # Show another pair
                    trajectories = sample_trajectories(_current_policy, time_horizon=30)  # TODO K???
                    traj_pair = random.sample(trajectories, 2)

                    comparison_html = compare_trajectories_html(traj_pair[0], traj_pair[1])
                    request.session['traj_pair'] = [traj_pair[0], traj_pair[1]]

                    context['show_comparison'] = True
                    context['comparison_html'] = comparison_html
                    context['preferences_collected'] = num_prefs
                    messages.info(request, f"Preference recorded! ({num_prefs}/10 collected)")
                else:
                    # Enough preferences collected, train the reward model
                    messages.success(request, f"Collected {num_prefs} preferences! Training reward model...")
                    _current_reward_net = train_with_human_preferences(request, context, _current_policy)
                    context['reward_net'] = True
                    request.session['comparison_mode'] = False
                    request.session['preferences'] = []
                    request.session.modified = True

        elif action == 'retrain':
            if _current_policy is not None and _current_reward_net is not None:
                retrain_policy(request, context, _current_policy, _current_reward_net)
                messages.success(request, "Policy retrained with RLHF!")
            else:
                messages.error(request, "Please complete Tasks 1 and 2 first!")

    return render(request, 'project5_base.html', context)


def sample_trajectory_check(request, context, policy):
    """ Generate a sample trajectory for visualizatio """

    if context['phase'] == 'task1':
        sample_traj = sample_trajectories(policy, K=1, time_horizon=20)[0]
        traj_html = trajectory_to_html(sample_traj, max_steps=5, title="Sample Trajectory After Training")

        context['training_results'].append({
            'task_name': 'Task 1: Initial Policy Training',
            'details': f'Policy trained with REINFORCE (500 trajectories)\n\nSample trajectory:\n{traj_html}'
        })
    elif context['phase'] == 'task2':
        sample_traj = sample_trajectories(policy, K=1, time_horizon=20)[0]
        traj_html = trajectory_to_html(sample_traj, max_steps=5, title="Sample Trajectory After Reward Model Training")

        # context['training_results'].append({
        #     'task_name': 'Task 2: Reward Model Training',
        #     'details': f'Policy evaluated after reward model training\n\nSample trajectory:\n{traj_html}'
        # })
    return None


def trajectory_simulation(request, context, policy):
    """Task 1: Train initial policy - modified to accept policy as parameter"""
    context['phase'] = 'task1'

    RL(policy, N_trajectories=3000)
    # Generate a sample trajectory for visualizatio
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

    # Add results to context
    organic_hits = sum(t["organic_hits"] for t in trajectories)
    cheese_hits = sum(t["cheese_hits"] for t in trajectories)
    context['training_results'].append({
        'task_name': 'Task 2: Reward Model Training',
        'details': f'Trained on {len(pairs)} preferences. Organic hits: {organic_hits}, Cheese hits: {cheese_hits}'
    })

    return reward_net


def train_with_human_preferences(request, context, policy):
    """Task 2b: Train reward model with collected human preferences"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get all collected preferences from session
    preferences = request.session.get('preferences', [])

    if not preferences:
        messages.error(request, "No preferences collected!")
        return None

    # Convert preferences to the format expected by train_reward_bt
    # We need a list of all unique trajectories and pair indices
    all_trajectories = []
    trajectory_map = {}  # To track unique trajectories
    pairs = []

    for pref in preferences:
        traj_a = pref['traj_a']
        traj_b = pref['traj_b']

        # Add trajectories to list if not already there
        # (In practice, we'd hash them properly, but for simplicity...)
        if id(traj_a) not in trajectory_map:
            trajectory_map[id(traj_a)] = len(all_trajectories)
            all_trajectories.append(traj_a)
        if id(traj_b) not in trajectory_map:
            trajectory_map[id(traj_b)] = len(all_trajectories)
            all_trajectories.append(traj_b)

        # Add pair with indices
        idx_a = trajectory_map[id(traj_a)]
        idx_b = trajectory_map[id(traj_b)]
        pairs.append((idx_a, idx_b, pref['preference']))

    # If we don't have enough unique trajectories, sample more
    if len(all_trajectories) < 20:
        extra_trajs = sample_trajectories(policy, K=20 - len(all_trajectories), time_horizon=30)
        all_trajectories.extend(extra_trajs)

    print(f"Training reward model with {len(pairs)} human preferences")

    # Train reward model
    reward_net = RewardNet()
    reward_net = train_reward_bt(
        reward_net, all_trajectories, pairs,
        epochs=20,
        lr=2e-3,
        batch_size=4,
        device=device
    )

    # Save the reward model
    torch.save(reward_net.state_dict(), os.path.join(SAVE_MODEL_PATH, "reward_net_human.pt"))
    print("Saved reward_net_human.pt")

    # Add results to context
    context['training_results'].append({
        'task_name': 'Task 2b: Reward Model (Human Preferences)',
        'details': f'Trained on {len(pairs)} human preference pairs'
    })

    return reward_net


def retrain_policy(request, context, policy, reward_net):
    """Task 3: Retrain policy - modified to accept policy and reward_net"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluate before training
    print("Evaluation BEFORE Task3 training:")
    before_stats = evaluate_policy_organic_avoidance(policy, K=200)
    print(before_stats)
    # 2) Retrain policy with learned reward + KL
    beta = 0.01  # Reduce from 0.003 to allow more learning

    # And these parameters in reinforce_with_learned_reward:
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

    # 3) Evaluate after training
    print("Evaluation AFTER Task3 training:")
    after_stats = evaluate_policy_organic_avoidance(trained_policy, K=200)
    print(after_stats)

    # Optionally save the trained policy
    torch.save(trained_policy.state_dict(), "policy_task3_rlhf.pt")
    print("Saved trained policy to policy_task3_rlhf.pt")

    # Update global policy
    global _current_policy
    _current_policy = trained_policy

    # Add results to context
    context['training_results'].append({
        'task_name': 'Task 3: RLHF Retraining',
        'details': f'Before: {before_stats["total_organic_hits"]} organic hits. After: {after_stats["total_organic_hits"]} organic hits.'
    })

    return None


# Load saved models on startup if they exist
def load_saved_states():
    global _current_policy, _current_reward_net

    if os.path.exists("current_policy.pt"):
        try:
            _current_policy = create_policy_network()
            _current_policy.load_state_dict(torch.load("current_policy.pt", map_location="cpu"))
            print("Loaded saved policy")
        except:
            pass

    if os.path.exists("reward_net_bt.pt"):
        try:
            _current_reward_net = RewardNet()
            _current_reward_net.load_state_dict(torch.load("reward_net_bt.pt", map_location="cpu"))
            print("Loaded saved reward net")
        except:
            pass


# Initialize on import
load_saved_states()
