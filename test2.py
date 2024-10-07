import cv2
import mss
import numpy as np
import pygame  # Import pygame for controller support
import gym
from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time

# Initialize pygame
pygame.init()
pygame.joystick.init()

# Check if the controller is connected
if pygame.joystick.get_count() == 0:
    raise Exception("No joystick (Xbox controller) found.")

# Use the first joystick (typically the Xbox controller)
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Function to display text on the screen
def display_on_monitor(frame, text="Selected Monitor"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (255, 0, 0)  # Red color
    thickness = 2

    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2

    cv2.putText(frame, text, (text_x, text_y), font, scale, color, thickness)
    return frame

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
        cv2.waitKey(1)

        return frame

# Preprocess the game frame (resize, grayscale, normalize)
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 84))
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Define possible actions (basic moves)
actions = ["move_left", "move_right", "attack", "dodge", "jump", "special"]

# Function to perform actions based on Xbox controller input
def perform_action(action):
    if action == "move_left":
        # Logic to move left (to be handled in your game)
        print("Moving left")
    elif action == "move_right":
        # Logic to move right (to be handled in your game)
        print("Moving right")
    elif action == "attack":
        # Logic to attack (to be handled in your game)
        print("Attacking")
    elif action == "dodge":
        # Logic to dodge (to be handled in your game)
        print("Dodging")
    elif action == "jump":
        # Logic to jump (to be handled in your game)
        print("Jumping")
    elif action == "special":
        # Logic for special move (to be handled in your game)
        print("Using special move")

# Define a custom Gym environment for RL
class WukongGameEnv(gym.Env):
    def __init__(self):
        super(WukongGameEnv, self).__init__()
        self.action_space = Discrete(len(actions))  # 6 possible actions
        self.observation_space = Box(low=0, high=1, shape=(84, 84, 1), dtype=np.float32)
    
    def reset(self):
        frame = capture_game_frame()
        processed_frame = preprocess_frame(frame)
        return processed_frame.reshape(84, 84, 1)
    
    def step(self, action):
        perform_action(actions[action])  # Perform action based on Xbox controller input
        time.sleep(0.05)  # Small delay to simulate real-time interaction

        next_frame = capture_game_frame()
        processed_frame = preprocess_frame(next_frame)

        # Define a reward function
        reward = self.calculate_reward(next_frame)
        
        done = False  # Update when needed

        return processed_frame.reshape(84, 84, 1), reward, done, {}

    def calculate_reward(self, frame):
        return 1.0  # Dummy reward for testing
    
    def render(self, mode='human'):
        pass

# Create the RL environment
env = DummyVecEnv([lambda: WukongGameEnv()])

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=1000)

# Save the model
model.save("ppo_wukong_ai")

# Load and test the trained agent
model = PPO.load("ppo_wukong_ai")

# Test the AI in-game
obs = env.reset()
for _ in range(1000):  # Test for 1000 steps
    # Capture Xbox controller input
    pygame.event.pump()  # Process the pygame event queue
    left_stick_x = joystick.get_axis(0)  # Left stick X-axis
    left_stick_y = joystick.get_axis(1)  # Left stick Y-axis
    button_a = joystick.get_button(0)  # A button for attack
    button_b = joystick.get_button(1)  # B button for dodge
    button_x = joystick.get_button(2)  # X button for jump
    button_y = joystick.get_button(3)  # Y button for special

    # Decide action based on controller input
    if left_stick_x < -0.5:  # Move left
        action = 0
    elif left_stick_x > 0.5:  # Move right
        action = 1
    elif button_a:  # Attack
        action = 2
    elif button_b:  # Dodge
        action = 3
    elif button_x:  # Jump
        action = 4
    elif button_y:  # Special move
        action = 5
    else:
        action = 0  # Default action

    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()

# Clean up and close OpenCV windows
cv2.destroyAllWindows()