import cv2
import mss
import numpy as np
import pyautogui
import gym
from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time

# Function to display text on the screen
def display_on_monitor(frame, text="Selected Monitor"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5  # Adjust scale down to 0.5 for smaller text
    color = (255, 0, 0)  # Red color
    thickness = 1  # Reduced thickness

    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2

    # Put the text on the frame
    cv2.putText(frame, text, (text_x, text_y), font, scale, color, thickness)

    # Optionally resize the frame if you want to display it in a smaller window
    frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))  # Resize to half the original size

    return frame_resized

# Capture the game frame
def capture_game_frame():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Capture from the second monitor (index 1)
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display text on the captured frame
        frame_with_text = display_on_monitor(frame, text="Selected Monitor")

        # Show the frame in an OpenCV window
        cv2.imshow("Monitor Indicator", frame_with_text)
        cv2.waitKey(1)  # Short wait to display the window

        return frame

# Preprocess the game frame (resize, grayscale, normalize)
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    resized_frame = cv2.resize(gray_frame, (84, 84))  # Resize for simplicity
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
    return normalized_frame

# Define possible actions (basic moves)
actions = ["move_left", "move_right", "attack", "dodge", "jump", "special"]

# Function to send actions to the game using PyAutoGUI
def perform_action(action):
    pass
    # if action == "move_left":
    #     pyautogui.keyDown('a')  # Move left
    #     pyautogui.keyUp('a')
    # elif action == "move_right":
    #     pyautogui.keyDown('d')  # Move right
    #     pyautogui.keyUp('d')
    # elif action == "attack":
    #     pyautogui.keyDown('j')  # Light attack
    #     pyautogui.keyUp('j')
    # elif action == "dodge":
    #     pyautogui.keyDown('space')  # Jump
    #     pyautogui.keyUp('space')
    # elif action == "jump":
    #     pyautogui.keyDown('k')  # Dodge
    #     pyautogui.keyUp('k')
    # elif action == "special":
    #     pyautogui.keyDown('l')  # Special move
    #     pyautogui.keyUp('l')

# Define a custom Gym environment for RL
class WukongGameEnv(gym.Env):
    def __init__(self):
        super(WukongGameEnv, self).__init__()
        self.action_space = Discrete(len(actions))  # 6 possible actions
        self.observation_space = Box(low=0, high=1, shape=(84, 84, 1), dtype=np.float32)  # 84x84 grayscale frame
    
    def reset(self):
        frame = capture_game_frame()  # Capture the game screen
        processed_frame = preprocess_frame(frame)  # Preprocess it
        return processed_frame.reshape(84, 84, 1)  # Return frame as observation
    
    def step(self, action):
        perform_action(actions[action])  # Perform action
        time.sleep(0.05)  # Small delay to simulate real-time interaction

        # Capture the next frame and preprocess
        next_frame = capture_game_frame()
        processed_frame = preprocess_frame(next_frame)

        # Define a reward function (can be improved with object detection)
        reward = self.calculate_reward(next_frame)
        
        # Simulate end of episode (boss health = 0, player dies, etc.)
        done = False  # Set 'done' to True when the episode should end

        return processed_frame.reshape(84, 84, 1), reward, done, {}

    # Simplified reward function for testing
    def calculate_reward(self, frame):
        # Dummy reward for testing
        return 1.0  # Give a constant reward (can be changed to detect boss damage, health, etc.)
    
    def render(self, mode='human'):
        pass  # Not necessary unless you want to render frames

# Create the RL environment
env = DummyVecEnv([lambda: WukongGameEnv()])

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model (replace with more timesteps for better results)
model.learn(total_timesteps=1000)

# Save the model
model.save("ppo_wukong_ai")

# Load and test the trained agent
model = PPO.load("ppo_wukong_ai")

# Test the AI in-game
obs = env.reset()
for _ in range(1000):  # Test for 1000 steps
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()

# Clean up and close OpenCV windows
cv2.destroyAllWindows()