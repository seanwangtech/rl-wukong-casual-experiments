
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
from nn_model23 import PPOnn

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
    # not normalized yet. need to normalize before feed the obs to model. 
    return env

# env = gym.make("PongNoFrameskip-v4", render_mode='rgb_array')
env_id = "PongNoFrameskip-v4"
env = make_env(env_id)
model = PPOnn(env.action_space.n).to(device)

hyperparameters = {
    "episodes":int(1e9),
    "batch_size":32, 
    "update_epochs":4,
    "learning_rate":2.5e-4,
    "gamma":0.99,
    "lam":0.95,
    "clip_epsilon":0.2,
    "dynamic_batch_size":True, 
    "min_num_minibatch_per_epoch":16 # the nums of minibatch may less than 16 when the episode length is short, because it need to guarantee mini_batch_size.  
}
optimizer = optim.Adam(model.parameters(), lr = hyperparameters['learning_rate']) 
agent = PPOAgent(
    action_space=env.action_space, 
    model=model, 
    optimizer=optimizer, 
    gamma= hyperparameters['gamma'], 
    lam=hyperparameters['lam'], 
    clip_epsilon=hyperparameters['clip_epsilon'], 
    device=device
)


writer = SummaryWriter(f"runs/{env_id}-{time.strftime('%Y%m%d-%H_%M_%S',time.localtime())}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % (
        "\n".join([f"|{key}|{value}|" for key, value in 
                   list(hyperparameters.items())
                   + [(key, repr(value)) for key, value in vars(agent).items()]
                   ])
    ),
)
writer.add_text("agent-vars",repr(vars(agent)))

def get_dynamic_batch_size(nth_episode, episode_length):
    """
    Due to the length of episode is not fixed, if the mini-batch size is fixed, PPO policy update for each iteration will not be stable.
    For example, if the episode length is short, a certain iteration will update policy a small number of times, let say 16 times.
    Once the episode length is long, the policy will be updated a lot more times in one iteration, let say 160 times, which result in overfitting to a specific episode, result unstable.
    We need to guarantee the policy shouldn't be updated to much within each policy iteration cycle.
    Use dynamic mini-batch size to achieve two purpose:
    1. initially, the batch size is small to ensure a quick training. Small batch is fast but high variance/unstable. we want it fast and high variance is acceptable initially. 
    2. Following, the batch size will increase to ensure a stable finaly result. Mitigate the affect of high variance of episode length. 
    3. finally, the min_num_minibatch_per_epoch will decide 4 mini-batch per epoch, each policy iteration will only update weights 4 times per epoch, avoid overfitting. 
    """
    mini_batch_size = hyperparameters['batch_size']
    mini_num_minibatch = hyperparameters['min_num_minibatch_per_epoch']
    batch_size = mini_batch_size + nth_episode # increase batch_size by 1 for each episode
    if(episode_length <= mini_batch_size* mini_num_minibatch):
        return mini_num_minibatch
    if(episode_length  < batch_size * mini_num_minibatch):
        batch_size = (episode_length - 1)//mini_num_minibatch + 1
    return batch_size


def obs2stateTensor(obs, show=False):
    # obs = obs[:, 420:1500] # choose area for model input
    # state = cv2.resize(obs, (224, 224))
    
    state = obs
    if(show): 
        cv2.imshow('model input', state[3])
        cv2.waitKey(1)
    state = torch.FloatTensor(state)
    state = state/255.0
    # state = state.permute(2, 0, 1).contiguous()  # (H, W, C) -> C, H, W
    # state = (state - state.mean())/state.std() # normalize
    return state

t1 = time.time()
episodes = hyperparameters['episodes']
batch_size = hyperparameters['batch_size']
update_epochs = hyperparameters['update_epochs']
for episode in range(episodes):
    state, _ = env.reset()
    state = obs2stateTensor(state, show=False)

    next_done = False
    total_reward = 0
    
    frame_count = 0
    episode_steps = 0
    while not next_done:
        action, log_prob, value = agent.select_action(state)
        obs_img, reward, terminated, truncated,  info = env.step(action)
        if(True):
            # not pased
            total_reward += reward
            next_done = terminated or truncated
            # print('reward',reward)
            next_state = obs2stateTensor(obs_img, show=False)
            agent.remember(state, action, reward, log_prob, value, next_done)
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
    if(hyperparameters['dynamic_batch_size']):
        batch_size = get_dynamic_batch_size(episode, episode_steps)
    policy_loss, value_loss, entropy_loss, clipfrac = agent.optimize(epochs=update_epochs, batch_size=batch_size)
    
    print(f"Episode Done: {episode}, Total Reward: {total_reward:.2f}, policy loss: {policy_loss:.4f}, value loss: {value_loss:.4f}")
    writer.add_scalar("charts/episode_reward", total_reward, episode)
    writer.add_scalar("charts/episode_length", episode_steps, episode)
    writer.add_scalar("charts/batch_size", batch_size, episode)
    writer.add_scalar("losses/policy_loss", policy_loss, episode)
    writer.add_scalar("losses/value_loss", value_loss, episode)
    writer.add_scalar("losses/entropy_loss", entropy_loss, episode)
    writer.add_scalar("losses/clipfrac", clipfrac, episode)
    # save target model
    if(episode%10==0):
        folder = f'{os.path.dirname(__file__)}/trains/'
        os.makedirs(folder, exist_ok=True)
        torch.save(agent.model.state_dict(), f'{folder}/model_{episode}.pth')
    
cv2.destroyAllWindows()
env.close()