import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r"C:\Users\raksh\optimising-policy-gradients")


# Import your training functions from the respective modules.
from src import train_reinforce, train_ppo, train_trpo, train_a2c

def run_experiment(algorithm, optimizer_type, seed, num_episodes):
    # Set random seeds for reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Choose and run the training routine for the selected algorithm.
    if algorithm.lower() == 'reinforce':
        rewards = train_reinforce.train(optimizer_type, num_episodes=num_episodes)
    elif algorithm.lower() == 'ppo':
        rewards = train_ppo.train(optimizer_type, num_episodes=num_episodes)
    elif algorithm.lower() == 'trpo':
        rewards = train_trpo.train(optimizer_type, num_episodes=num_episodes)
    elif algorithm.lower() == 'a2c':
        rewards = train_a2c.train(optimizer_type, num_episodes=num_episodes)
    else:
        raise ValueError("Algorithm not recognized. Choose from: reinforce, ppo, trpo, a2c")
    return rewards

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DRL Policy Gradient Algorithms with Various Optimizers")
    parser.add_argument('--algorithm', type=str, default='reinforce',
                        help="Algorithm to run: reinforce, ppo, trpo, a2c")
    parser.add_argument('--optimizer', type=str, default='adam',
                        help="Optimizer to use: sgd, adam, rmsprop")
    parser.add_argument('--runs', type=int, default=5,
                        help="Number of independent runs (different seeds)")
    parser.add_argument('--episodes', type=int, default=500,
                        help="Number of episodes per run")
    args = parser.parse_args()
    
    all_rewards = []
    for run in range(args.runs):
        seed = 42 + run
        print(f"Run {run+1} with seed {seed}")
        rewards = run_experiment(args.algorithm, args.optimizer, seed, args.episodes)
        all_rewards.append(rewards)
    
    all_rewards = np.array(all_rewards)  # shape: (runs, episodes)
    avg_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    
    # Plot the average learning curve with standard deviation.
    episodes = np.arange(1, args.episodes + 1)
    plt.figure()
    plt.plot(episodes, avg_rewards, label='Average Reward')
    plt.fill_between(episodes, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.3)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(f"{args.algorithm.upper()} with {args.optimizer.upper()}")
    plt.legend()
    
    # Ensure the results directory exists.
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    filename = os.path.join(results_dir, f"{args.algorithm}_{args.optimizer}_avg_rewards.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved average learning curve to {filename}")
    
    # Print final performance metrics.
    final_rewards = all_rewards[:, -1]
    print(f"Final reward over {args.runs} runs: mean = {np.mean(final_rewards):.2f}, std = {np.std(final_rewards):.2f}")

if __name__ == '__main__':
    main()
