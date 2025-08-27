from django.shortcuts import render

# Create your views here.
from reinforcement_learning.policy_network import create_policy_network
from reinforcement_learning.trajectory_simulation import RL
from reinforcement_learning.rlhf import sample_trajectories, build_pairwise_preferences_sim, RewardNet, train_reward_bt
from reinforcement_learning.retraining_with_RLHF import evaluate_policy_organic_avoidance, reinforce_with_learned_reward
import torch
import numpy as np

def index(request):
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'trajectory':
            trajectory_simulation(request, context)

        elif action == 'train_sim':
            train_with_rlhf_sim(request, context)

        elif action == 'train_human':
            train_with_rlhf_human(request, context)

        elif action == 'retrain':
            retrain_policy(request, context)

    return render(request, 'project5_base.html', context)

def trajectory_simulation(request, context):
    policy = create_policy_network()
    RL(policy)
    context['policy'] = policy
    return None

def train_with_rlhf_sim(request, context):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trajectories = sample_trajectories(context['policy'], K=100, time_horizon=50)  # Increased from 80

    pairs = build_pairwise_preferences_sim(trajectories, provider="sim", max_pairs=120)

    reward_net = RewardNet()
    reward_net = train_reward_bt(
        reward_net, trajectories, pairs,
        epochs=20,  # More epochs
        lr=2e-3,  # Higher learning rate
        batch_size=6,  # Slightly smaller batch
        device=device
    )

    torch.save(reward_net.state_dict(), "reward_net_bt.pt")
    print("Saved reward_net_bt.pt")
    return None

def train_with_rlhf_human(request, context):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trajectories = sample_trajectories(context['policy'], K=100, time_horizon=50)  # Increased from 80
    prefs = []
    N = len(trajectories)

    pairs_created = 0
    for i in range(N):
        for j in range(i + 1, N):
            if pairs_created >= 120:
                break

            # Prioritize pairs where organic hits differ significantly
            oa, ob = trajectories[i]["organic_hits"], trajectories[j]["organic_hits"]
            if abs(oa - ob) > 0 or np.random.random() < 0.3:  # Include some random pairs too
                y = choose(trajectories[i], trajectories[j])
                #TODO let the user choose whether trajectories[i] or trajectories[j] - we need some nice print
                prefs.append((i, j, y)) #y=1 if option trajectories[i] y = 0 if trajectories[j]
                pairs_created += 1

        if pairs_created >= 120:
            break


    reward_net = RewardNet()
    reward_net = train_reward_bt(
        reward_net, trajectories, prefs,
        epochs=20,  # More epochs
        lr=2e-3,  # Higher learning rate
        batch_size=6,  # Slightly smaller batch
        device=device
    )

    torch.save(reward_net.state_dict(), "reward_net_bt.pt")
    print("Saved reward_net_bt.pt")
    return None

def retrain_policy(request, context):
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

    return None
