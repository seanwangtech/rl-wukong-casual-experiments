import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import frame_stack,AtariPreprocessing
import numpy as np
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
def make_env(env_id, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)
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
    # not normalized yet. need to normalize before feed the obs to model. 
    return env
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_script_dir)

env = gym.make("PongNoFrameskip-v4", render_mode='human')
# env = make_env("PongNoFrameskip-v4", render_mode='human')
env = AtariPreprocessing(env, grayscale_obs = True,scale_obs=True, frame_skip=4)
env = frame_stack.FrameStack(env, 4)
from nn_model23 import PPOnn
model = PPOnn(env.action_space.n)

wei = torch.load("demo-1/model_1040.pth", weights_only=True)
model.load_state_dict(wei)
for param in model.parameters():
    print(f'number of parameters:{param.numel(): 12d} {param.shape}')
    

state, _ = env.reset()
total_rewards = []
for episode in range(100):
    total_reward = 0
    next_done = False
    while not next_done:
        with torch.no_grad():
            logits, value = model(torch.FloatTensor(state).unsqueeze(0))
        # Sample action from the policy
        action_distribution = torch.distributions.Categorical(logits=logits)
        action = action_distribution.sample()
        state, reward, terminated, truncated,  info = env.step(action)
        next_done = terminated or truncated
        total_reward += reward

    state, _ = env.reset()
    total_rewards.append(total_reward)
    print(f'{episode}: total_reward: {total_reward}, average_reward: {np.mean(total_rewards)}')
    
    