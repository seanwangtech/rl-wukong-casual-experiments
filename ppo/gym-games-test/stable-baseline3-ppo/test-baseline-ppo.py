import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Set up the Atari environment
env_name = "ALE/Kaboom-v5"
env = make_atari_env(env_name, n_envs=4, seed=0)  # Parallelize over 4 environments for efficiency
env = VecFrameStack(env, n_stack=4)  # Stack 4 frames to provide temporal information to the agent

# Check the observation and action spaces
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Create the PPO model
model = PPO(
    "CnnPolicy",  # Use a CNN policy since observations are images
    env,
    verbose=1,
    tensorboard_log="./ppo_kaboom_tensorboard/",  # Path to log training stats for TensorBoard
    n_steps=128,  # Rollout steps per environment
    batch_size=256,  # Batch size for updates
    n_epochs=4,  # Number of epochs per update
    gamma=0.99,  # Discount factor
    learning_rate=2.5e-4,  # Learning rate
    clip_range=0.1,  # Clipping for PPO
)

# Train the model
time_steps = 1_000_000  # Number of steps to train the model for
# model.learn(total_timesteps=time_steps)

# # Save the trained model
# model.save("ppo_kaboom")

# To load the model later and test it
model = PPO.load("ppo_kaboom")

# Run a test episode
obs = env.reset()
for _ in range(10000):  # Run for a fixed number of frames or until done
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()