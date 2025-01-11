
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
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


def make_env(env_id):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env

# env = gym.make("PongNoFrameskip-v4", render_mode='rgb_array')
env_id = "PongNoFrameskip-v4"
env = make_env(env_id)
model = PPOnn(env.action_space.n).to(device)
optimizer = optim.Adam(model.parameters(), lr=2.5e-4)

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
batch_size = 256
update_epochs = 10


writer = SummaryWriter(f"runs/{env_id}-{time.strftime('%Y%m%d-%H_%M_%S',time.localtime())}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % (
        "\n".join([f"|{key}|{value}|" for key, value in 
                   [('batch_size', batch_size), ("update_epochs", update_epochs)]
                   + [(key, repr(value)) for key, value in vars(agent).items()]
                   ])
    ),
)
writer.add_text("agent-vars",repr(vars(agent)))




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
    state = obs2stateTensor(state, show=False)

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
    writer.add_scalar("charts/episode_reward", total_reward, episode)
    writer.add_scalar("charts/episode_length", episode_steps, episode)
    writer.add_scalar("losses/policy_loss", policy_loss, episode)
    writer.add_scalar("losses/value_loss", value_loss, episode)
    # save target model
    if(episode%10==0):
        folder = f'{os.path.dirname(__file__)}/trains/'
        os.makedirs(folder, exist_ok=True)
        torch.save(agent.model.state_dict(), f'{folder}/model_{episode}.pth')
    
cv2.destroyAllWindows()
env.close()