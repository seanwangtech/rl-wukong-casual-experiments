
import random
from collections import deque
import torch

class DDQNAgent:
    def __init__(self, action_space, policy_net, target_net, optimizer, loss_fn, gamma=0.99, device='cpu'):
        
        self.action_space = action_space
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.memory = deque(maxlen=100000)
        self.policy_net = policy_net
        self.target_net = target_net  # Set target model for DDQN
        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()
        self.epsilon = 1.0  # Epsilon-greedy action selection
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.device = device
        print(self.device)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def select_action(self, state):
        if random.random() <= self.epsilon:
            return self.action_space.sample() # Explore
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
        return q_values.argmax().item()   # Exploit

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        # Double DQN: Use policy net to choose action, target net to calculate Q-value of that action
        with torch.no_grad():
            # Get the best action from the policy network
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # Use target network to get Q-values for these actions
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
            