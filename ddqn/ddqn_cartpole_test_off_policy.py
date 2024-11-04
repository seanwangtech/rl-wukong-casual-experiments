import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from nn_model import DQNEfficientNet
from ddqn_agent import DDQNAgent
import gymnasium as gym
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env =  gym.make('CartPole-v1', render_mode='human')
policy_net = DQN(env.observation_space.shape[0],env.action_space.n).to(device)
target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
loss_fn = F.mse_loss
# Assuming `model` is an instance of DQNEfficientNet and learning_rate is defined
learning_rate = 1e-6

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(policy_net.parameters(),lr=0.001)
# for param in model.features.parameters():
#     param.requires_grad = False  # Freeze pre-trained layers

agent = DDQNAgent(env.action_space, 
                  policy_net=policy_net,
                  target_net=target_net,
                  optimizer=optimizer, loss_fn=loss_fn, device=device)

episodes = int(1e9)
batch_size = 32

def obs2stateTensor(obs, show=False):
    return torch.FloatTensor(obs)
losses = []
for episode in range(episodes):
    state, _ = env.reset()
    state = obs2stateTensor(state, show=True)

    done = False
    total_reward = 0
    
    frame_count = 0
    episode_steps = 0
    t1 = time.time()
    while not done:
        action = agent.select_action(state)
        obs, reward, done, trancated,info  = env.step(action)
        if(True):
            # not pased
            total_reward += reward
            # reward = 0 if done else 0.1 # works greatly
            # reward = -1 if done else 0 # works greatly
            # print('reward',reward)
            next_state = obs2stateTensor(obs, show=True)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_steps += 1
        
        
        if time.time() - t1 > 10:
            FPS = frame_count / (time.time() - t1)
            frame_count = 0
            t1 = time.time()
            img = obs 
            # env.draw_areas(img)
            print(f'    Ep: {episode}, FPS: {FPS:.2f}, total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}')
        frame_count += 1
        time.sleep(0.005)
        
    # Update network after episode
    for i in range(episode_steps):
        loss = agent.replay(batch_size)
        agent.update_epsilon()
        losses.append(loss)
        # if i%20==0:
        #     agent.update_target_net()
    episode_steps = 0
    # Update target network after every episode
    agent.update_target_net()
    print(f"Episode Done: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    plt.plot(losses)
    plt.yscale('log')
    plt.draw()
    plt.pause(0.01)

cv2.destroyAllWindows()