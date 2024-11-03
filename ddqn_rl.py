import gymnasium as gym
import cv2
import numpy as np
import torch
import pyautogui
from gymnasium import spaces
from pynput import keyboard

class BlackMythWukongEnv(gym.Env):
    def __init__(self):
        super(BlackMythWukongEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)  # 0: dodge, 1: use_gourd, 2: light_attack, 3: heavy_attack
        self.paused = True
        self.screen_width = 1920  # Set according to your screen resolution
        self.screen_height = 1080

        # Listener to pause the environment
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def on_key_press(self, key):
        if key == keyboard.KeyCode.from_char('0'):
            self.paused = not self.paused

    def reset(self):
        # Reset the game state (if possible), here we'll just return the initial observation
        return self._get_observation()

    def step(self, action):
        if self.paused:
            return None, 0, False, {}

        # Map actions to game controls
        if action == 0:
            pyautogui.press('space')       # Dodge
        elif action == 1:
            pyautogui.press('r')           # Use Gourd
        elif action == 2:
            pyautogui.click(button='left')  # Light Attack
        elif action == 3:
            pyautogui.click(button='right') # Heavy Attack

        # Obtain new observation and calculate reward
        observation = self._get_observation()
        print(observation.shape)
        reward = self._calculate_reward(observation)
        done = False  # Define condition for end of game if possible

        return observation, reward, done, {}

    def _get_observation(self):
        # Capture screenshot of the game window
        screenshot = pyautogui.screenshot(region=(0, 0, self.screen_width, self.screen_height))
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, (224, 224))
        return resized_img

    def _calculate_reward(self, observation):
        # Analyze the screenshot to determine rewards based on health bars
        boss_health = self._extract_boss_health(observation)
        player_health = self._extract_player_health(observation)
        reward = (player_health - boss_health) * 10  # Example reward calculation
        return reward

    def _extract_boss_health(self, observation):
        # Use image processing to extract the boss's health from the observation
        # Placeholder; adjust this to use the actual boss health bar region
        return 100  # Example fixed value

    def _extract_player_health(self, observation):
        # Use image processing to extract the player's health from the observation
        # Placeholder; adjust this to use the actual player health bar region
        return 100  # Example fixed value
    

# from efficientnet_pytorch import EfficientNet
import timm
import torch.nn as nn

class DQNEfficientNet(nn.Module):
    def __init__(self, num_actions):
        super(DQNEfficientNet, self).__init__()
        # self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        
        # Remove the last fully connected layer by slicing (EfficientNet's output features are 1280)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # Custom fully connected layers for RL
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x
    
import random
from collections import deque

class DDQNAgent:
    def __init__(self, action_space, model, optimizer, loss_fn, gamma=0.99):
        self.action_space = action_space
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.memory = deque(maxlen=2000)
        self.target_model = model  # Set target model for DDQN
        self.epsilon = 1.0  # Epsilon-greedy action selection
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.choice(range(self.action_space.n))
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                target += self.gamma * self.target_model(next_state).max(1)[0].item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0).to(device))
            target_f[0][action] = target
            loss = self.loss_fn(target_f, torch.FloatTensor(target).unsqueeze(0).to(device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
            
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = BlackMythWukongEnv()
model = DQNEfficientNet(env.action_space.n)
loss_fn = F.mse_loss
# Assuming `model` is an instance of DQNEfficientNet and learning_rate is defined
learning_rate = 1e-4

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam([
    {'params': model.features.parameters(), 'lr': 1e-5},  # Low LR for pre-trained backbone
    {'params': model.fc.parameters(), 'lr': 1e-3}  # Higher LR for classifier
],lr=0.001)
# for param in model.features.parameters():
#     param.requires_grad = False  # Freeze pre-trained layers

agent = DDQNAgent(env.action_space, model, optimizer, loss_fn)

episodes = 1000
batch_size = 32

for episode in range(episodes):
    state = env.reset()
    state = torch.FloatTensor(state).to(device)
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.replay(batch_size)
        # Display real-time window
        cv2.imshow('Game Analysis', state)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Episode: {episode}, Total Reward: {total_reward}")

cv2.destroyAllWindows()