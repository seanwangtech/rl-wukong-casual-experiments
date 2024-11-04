import cv2
import numpy as np
import torch

import random
from collections import deque

class DDQNAgent:
    def __init__(self, action_space, policy_net, target_net, optimizer, loss_fn, gamma=0.99, device='cpu'):
        
        self.action_space = action_space
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.memory = deque(maxlen=2000)
        self.policy_net = policy_net
        self.target_net = target_net  # Set target model for DDQN
        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()
        self.epsilon = 0.4  # Epsilon-greedy action selection
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.device = device

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
            
            
import torch.nn.functional as F
import torch.optim as optim
import time
from wukong_env import BlackMythWukongEnv
from nn_model import DQNEfficientNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = BlackMythWukongEnv()
policy_net = DQNEfficientNet(env.action_space.n)
target_net = DQNEfficientNet(env.action_space.n)
loss_fn = F.mse_loss
# Assuming `model` is an instance of DQNEfficientNet and learning_rate is defined
learning_rate = 1e-4

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam([
    {'params': policy_net.features.parameters(), 'lr': 1e-5},  # Low LR for pre-trained backbone
    {'params': policy_net.fc.parameters(), 'lr': 1e-3}  # Higher LR for classifier
],lr=0.001)
# for param in model.features.parameters():
#     param.requires_grad = False  # Freeze pre-trained layers

agent = DDQNAgent(env.action_space, 
                  policy_net=policy_net,
                  target_net=target_net,
                  optimizer=optimizer, loss_fn=loss_fn, device=device)

episodes = int(1e9)
batch_size = 4

def obs2stateTensor(obs, device, show=False):
    obs = obs[:, 420:1500] # choose area for model input
    state = cv2.resize(obs, (224, 224))
    if(show): 
        cv2.imshow('model input', cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
    state = torch.FloatTensor(state)
    state = state.permute(2, 0, 1)  # (H, W, C) -> C, H, W
    return state

t1 = time.time()
for episode in range(episodes):
    state = env.reset()
    state = obs2stateTensor(state, device, show=True)

    done = False
    total_reward = 0
    
    frame_count = 0
    while not done:
        action = agent.select_action(state)
        obs_img, reward, done, info = env.step(action)
        if(not info['pasued']):
            # not pased
            total_reward += reward
            # print('reward',reward)
            next_state = obs2stateTensor(obs_img, device, show=True)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(batch_size)
            agent.update_epsilon()
            if(frame_count%20==0):
                agent.update_target_net()
            total_reward += reward
        
        
        if time.time() - t1 > 1:
            FPS = frame_count / (time.time() - t1)
            frame_count = 0
            t1 = time.time()
            img = obs_img 
            env.draw_areas(img)
            print(f'[{"Paused" if info["pasued"] else "Trainning"}] Ep: {episode}, FPS: {FPS:.2f}, total reward: {total_reward:.2f}, \
Epsilon: {agent.epsilon:.2f}, \
wukong (H, M, S): {(info["wukong_health"], info["wukong_mana"], info["wukong_stamina"])}, boss: {info["boss_health"]}')
            cv2.imshow('Game Analysis', cv2.cvtColor(cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(5):
                break
        frame_count += 1
        time.sleep(0.005)
    time.sleep(0.2)
    if(frame_count>5):
        print(f"Episode Done: {episode}, Total Reward: {total_reward}")

cv2.destroyAllWindows()