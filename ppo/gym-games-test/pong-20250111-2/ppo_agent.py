import random
import torch
import torch.nn.functional as F
from collections import deque
import numpy as np

class PPOAgent:
    def __init__(self, action_space, model, optimizer, gamma=0.99, lam=0.95, clip_epsilon=0.2, device='cpu'):
        self.action_space = action_space
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.lam = lam  # Lambda for GAE
        self.clip_epsilon = clip_epsilon  # Clipping range for policy loss
        self.device = device

        # Stores episode data: states, actions, log probabilities, rewards, values, and dones
        self.memory = []

    def remember(self, state, action, reward, log_prob, value, done):
        """Store transition in memory."""
        self.memory.append((state, action, reward, log_prob, value, done))
    
    def select_action(self, state):
        """Select an action using the current policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(state)

        # Sample action from the policy
        action_distribution = torch.distributions.Categorical(logits=logits)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)

        return action.item(), log_prob, value.item()

    def compute_gae(self, rewards, values, dones):
        values = list(values) + [0]  # Convert values to a list before concatenation
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.append(gae)
        advantages.reverse()
        return advantages

    def process_memory(self):
        """Prepare memory for optimization."""
        states, actions, rewards, log_probs, values, dones = zip(*self.memory)
        with torch.no_grad():
            returns = []
            G = 0
            # Compute returns for each timestep
            for reward, done in zip(reversed(rewards), reversed(dones)):
                G = reward + self.gamma * G * (1 - done)
                returns.append(G)
            returns.reverse()
            
            # Compute advantages using GAE
            advantages = self.compute_gae(rewards, values, dones)
            # Normalize advantages for training stability
            advantages = torch.FloatTensor(advantages).to(self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Prepare tensors
            states = torch.stack(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            old_log_probs = torch.FloatTensor(log_probs).to(self.device)

        return states, actions, returns, old_log_probs, advantages

    def optimize(self, epochs=4, batch_size=32):
        """Optimize policy and value networks using PPO update."""
        states, actions, returns, old_log_probs, advantages = self.process_memory()

        for _ in range(epochs):
            inds = np.arange(len(states))
            np.random.shuffle(inds)
            for i in range(0, len(states), batch_size):
                # Sample a mini-batch
                batch_indices = inds[i: i + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Calculate new log probabilities and entropy
                logits,values = self.model(batch_states)
                action_distribution = torch.distributions.Categorical(logits=logits)
                new_log_probs = action_distribution.log_prob(batch_actions)
                entropy = action_distribution.entropy().mean()

                # Policy loss with clipping
                ratio = (new_log_probs - batch_old_log_probs).exp()
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                # Value loss (Mean Squared Error between returns and values)
                values = values.squeeze(-1)
                value_loss = F.mse_loss(values, batch_returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.1 * entropy

                # Update networks
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear memory after each optimization step
        self.memory = []

        return policy_loss.item(), value_loss.item()
    
  