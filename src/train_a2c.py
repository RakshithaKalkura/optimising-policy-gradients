import torch
import torch.nn as nn
import torch.optim as optim
from src.environment import create_env
from src.policy_network import PolicyNetwork
from src.utils import save_training_plot

# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train(optimizer_type='adam', num_episodes=500, gamma=0.99, learning_rate=1e-3):
    env = create_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize policy and value networks
    policy = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)
    
    # Set up the optimizers for both networks
    if optimizer_type.lower() == 'sgd':
        optimizer_policy = optim.SGD(policy.parameters(), lr=learning_rate)
        optimizer_value = optim.SGD(value_net.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'adam':
        optimizer_policy = optim.Adam(policy.parameters(), lr=learning_rate)
        optimizer_value = optim.Adam(value_net.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer_policy = optim.RMSprop(policy.parameters(), lr=learning_rate)
        optimizer_value = optim.RMSprop(value_net.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer type. Choose from: sgd, adam, rmsprop")
    
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        log_probs = []
        rewards = []
        values = []
        done = False
        
        while not done:
            # Ensure the state is a float tensor (float32)
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)
            value = value_net(state_tensor)
            values.append(value)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            log_probs.append(log_prob)
            
            next_state, reward, done, truncated, info = env.step(action.item())
            rewards.append(reward)
            state = next_state
        
        # Compute discounted returns and convert to float32
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize the returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        values = torch.cat(values)  # values from critic
        log_probs = torch.stack(log_probs)
        
        # Compute advantage
        advantages = returns - values.squeeze()
        
        # Calculate losses for policy and value networks
        policy_loss = - (log_probs * advantages.detach()).mean()
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        loss = policy_loss + value_loss
        
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()
        loss.backward()
        optimizer_policy.step()
        optimizer_value.step()
        
        total_reward = sum(rewards)
        rewards_history.append(total_reward)
        if episode % 50 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")
    
    save_training_plot(rewards_history, "results/a2c_rewards.png")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        optimizer_type = sys.argv[1]
    else:
        optimizer_type = 'adam'
    train(optimizer_type=optimizer_type)
