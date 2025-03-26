import torch
import torch.optim as optim
from src.environment import create_env
from src.policy_network import PolicyNetwork
from src.utils import save_training_plot

def train(optimizer_type='adam', num_episodes=500, gamma=0.99, epsilon=0.2, learning_rate=1e-3):
    env = create_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = PolicyNetwork(state_dim, action_dim)
    
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(policy.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(policy.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer type")
    
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            next_state, reward, done, truncated, info = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        old_log_probs = torch.stack(log_probs).detach()
        current_log_probs = torch.stack(log_probs)
        
        ratio = torch.exp(current_log_probs - old_log_probs)
        obj1 = ratio * returns
        obj2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * returns
        loss = -torch.min(obj1, obj2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_reward = sum(rewards)
        rewards_history.append(total_reward)
        if episode % 50 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")
    
    save_training_plot(rewards_history, "results/ppo_rewards.png")
