
import gymnasium as gym
from gymnasium.wrappers import frame_stack,AtariPreprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
import pandas as pd

from ppo_agent import PPOAgent
from nn_model20 import PPOnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('torch device:',device)
env = gym.make("PongNoFrameskip-v4", render_mode='rgb_array')
env = AtariPreprocessing(env, grayscale_obs = True,scale_obs=True, frame_skip=4)
env = frame_stack.FrameStack(env, 4)
action_dim = env.action_space.n
model = PPOnn(action_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

agent = PPOAgent(
    action_space=env.action_space, 
    model=model, 
    optimizer=optimizer, 
    gamma=0.99, 
    lam=0.95, 
    clip_epsilon=0.2, 
    device=device
)






episodes = int(1e9)
batch_size = 32
update_epochs = 10
policy_losses = []
value_losses = []
total_rewards = []
plt.figure(figsize=(12, 4))
pltsubs = (plt.subplot(1, 3, 1), plt.subplot(1, 3, 2), plt.subplot(1, 3, 3))
pltsubs[0].set_title('policy_loss')
pltsubs[1].set_yscale('log')
pltsubs[1].set_title('value_loss')
pltsubs[2].set_title('reward')
def obs2stateTensor(obs, show=False):
    # obs = obs[:, 420:1500] # choose area for model input
    # state = cv2.resize(obs, (224, 224))
    
    state = obs
    if(show): 
        cv2.imshow('model input', state[3])
        cv2.waitKey(1)
    state = torch.FloatTensor(state)
    # state = state.permute(2, 0, 1).contiguous()  # (H, W, C) -> C, H, W
    # state = (state - state.mean())/state.std() # normalize
    return state

t1 = time.time()
for episode in range(episodes):
    state, _ = env.reset()
    state = obs2stateTensor(state, show=True)

    done = False
    total_reward = 0
    
    frame_count = 0
    episode_steps = 0
    while not done:
        action, log_prob, value = agent.select_action(state)
        obs_img, reward, done, truncated,  info = env.step(action)
        if(True):
            # not pased
            total_reward += reward
            # print('reward',reward)
            next_state = obs2stateTensor(obs_img, show=False)
            agent.remember(state, action, reward, log_prob, value, done or truncated)
            state = next_state
            # agent.replay(batch_size)
            # agent.update_epsilon()
            # if(frame_count%20==0):
            #     agent.update_target_net()
            episode_steps += 1
        if time.time() - t1 > 1:
            FPS = frame_count / (time.time() - t1)
            frame_count = 0
            t1 = time.time()
            img = obs_img 
            print(f'Ep: {episode}, FPS: {FPS:.2f}, total reward: {total_reward:.2f}')
        frame_count += 1
        # time.sleep(0.005)
    
    print('episode done, Training ...')
    policy_loss, value_loss = agent.optimize(epochs=update_epochs, batch_size=batch_size)
    
    print(f"Episode Done: {episode}, Total Reward: {total_reward:.2f}, policy loss: {policy_loss:.4f}, value loss: {value_loss:.4f}")
    total_rewards.append(total_reward)
    policy_losses.append(policy_loss)
    value_losses.append(value_loss)
    # save target model
    if(episode%10==0):
        folder = f'{os.path.dirname(__file__)}/trains/'
        os.makedirs(folder, exist_ok=True)
        torch.save(agent.model.state_dict(), f'{folder}/model_{episode}.pth')
        pd.DataFrame({
                'policy_loss': policy_losses,
                'value_loss': value_losses,
                'reward': total_rewards
                }).to_csv(f'{folder}/log_{episode}.csv')
    if(episode>20):
        print(f"averge plicy_loss:{np.mean(policy_losses[-21:-1])}, averge value_loss:{np.mean(value_losses[-21:-1])}, average reward:{np.mean(total_rewards[-21:-1])}")
    if(episode % int(episode//100+1) == 0): # limit plot time to reduce time spended on plot graph
        # draw graph
        pltsubs[0].plot(policy_losses)
        pltsubs[1].plot(value_losses)
        pltsubs[2].plot(total_rewards)
        plt.draw()
        plt.pause(0.01)

cv2.destroyAllWindows()
env.close()