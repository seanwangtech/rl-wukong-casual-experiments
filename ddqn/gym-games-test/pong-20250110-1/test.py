import gymnasium as gym
import numpy as np
import cv2
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO

# Function to preprocess the frame (resize to 84x84)
def preprocess_observation(obs):
    # Convert observation to grayscale, resize to 84x84, and normalize
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    data= np.expand_dims(resized_frame, axis=0)  # Add channel dimension
    std = data.std(axis=(1, 2), keepdims=True)
    mean = data.mean(axis=(1, 2), keepdims=True)
    return (data - mean) / std

# Wrap the environment for custom preprocessing
class PreprocessedEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Update the observation space to 84x84 with a single channel
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.float32
        )

    def observation(self, obs):
        return preprocess_observation(obs)

# Load the pretrained model from Hugging Face Hub
checkpoint = load_from_hub(
    repo_id="maxstahl/ppo-Pongv5",
    filename="ppo.zip",
)
model = PPO.load(checkpoint)

# Create and wrap the Pong environment
env = gym.make("ALE/Pong-v5", render_mode="human")  # Use render_mode="human" to display the game
env = PreprocessedEnv(env)

# Play the game using the model
obs, _ = env.reset()
while True:
    # Predict the action to take
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    # Check for game over
    if terminated or truncated:
        obs, _ = env.reset()