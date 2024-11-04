import gymnasium as gym
import cv2
import numpy as np
import torch
import pyautogui
from gymnasium import spaces
from pynput import keyboard

class BlackMythWukongEnv(gym.Env):
    def __init__(self,
                 wukong_health_bar_region=(208, 982, 600, 992), # (x1, y1, x2, y2), for 1920x1080, the area is (208, 982, 600, 992)
                 wukong_mana_bar_region=(208, 1002, 600, 1008), # (x1, y1, x2, y2), for 1920x1080, the area is (208, 1002, 600, 1008)
                 wukong_stamina_bar_region=(208, 1016, 600, 1021), # (x1, y1, x2, y2), for 1920x1080, the area is (208, 1016, 600, 1021)
                 boss_health_bar_region=(760, 912, 1172, 922), # (x1, y1, x2, y2), for 1920x1080, the area is (760, 912, 1172, 922)
                 wukong_health_color_lowerb = (170, 70, 70),  # Color is RGB
                 wukong_health_color_upperb = (230, 230, 230),
                 wukong_mana_color_lowerb = (45, 80, 130),
                 wukong_mana_color_upperb = (85, 140, 210),
                 wukong_stamina_color_lowerb = (110, 130, 75),
                 wukong_stamina_color_upperb = (200, 165, 110),
                 boss_health_color_lowerb = (170, 170, 170),
                 boss_health_color_upperb = (230, 230, 230),
                 ):
        super(BlackMythWukongEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 0: dodge, 1: use_gourd, 2: light_attack, 3: heavy_attack
        self.paused = True
        self.screen_width = 1920  # Set according to your screen resolution
        self.screen_height = 1080
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype=np.uint8)

        self.wukong_health_bar_region = wukong_health_bar_region
        self.wukong_mana_bar_region = wukong_mana_bar_region
        self.wukong_stamina_bar_region = wukong_stamina_bar_region
        self.boss_health_bar_region = boss_health_bar_region

        self.wukong_health_color_lowerb = wukong_health_color_lowerb
        self.wukong_health_color_upperb = wukong_health_color_upperb
        self.wukong_mana_color_lowerb = wukong_mana_color_lowerb
        self.wukong_mana_color_upperb = wukong_mana_color_upperb
        self.wukong_stamina_color_lowerb = wukong_stamina_color_lowerb
        self.wukong_stamina_color_upperb = wukong_stamina_color_upperb
        self.boss_health_color_lowerb = boss_health_color_lowerb
        self.boss_health_color_upperb = boss_health_color_upperb
        

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
        if not self.paused:
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
        reward = self._calculate_reward(observation)
        done = False  # Define condition for end of game if possible

        return observation, reward, done, {'pasued': self.paused,
                                            'wukong_health': self._extract_wukong_health(observation),
                                            'wukong_mana': self._extract_wukong_mana(observation),
                                            'wukong_stamina': self._extract_wukong_stamina(observation),
                                            'boss_health': self._extract_boss_health(observation)}

    def _get_observation(self):
        # Capture screenshot of the game window
        screenshot = pyautogui.screenshot(region=(0, 0, self.screen_width, self.screen_height))
        img = np.array(screenshot)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # no need to do covert, the pyautogui screenshot already in RGB
        # resized_img = cv2.resize(img, (224*2, 224*2))
        return img

    def _calculate_reward(self, observation):
        # Analyze the screenshot to determine rewards based on health bars
        boss_health = self._extract_boss_health(observation)
        player_health = self._extract_wukong_health(observation)
        reward = (player_health - boss_health) * 10  # Example reward calculation
        return reward

    def _extract_boss_health(self, observation):
        # Use image processing to extract the boss's health from the observation
        # Placeholder; adjust this to use the actual boss health bar region
        return self._pixel_count(observation, self.boss_health_bar_region,
                                 self.boss_health_color_lowerb, self.boss_health_color_upperb)

    def _extract_wukong_health(self, observation):
        # Use image processing to extract the player's health from the observation
        # Placeholder; adjust this to use the actual player health bar region
        return self._pixel_count(observation, self.wukong_health_bar_region,
                                 self.wukong_health_color_lowerb, self.wukong_health_color_upperb)
        
    def _extract_wukong_mana(self, observation):
        # Use image processing to extract the player's mana from the observation
        # Placeholder; adjust this to use the actual player mana bar region
        return self._pixel_count(observation, self.wukong_mana_bar_region,
                                 self.wukong_mana_color_lowerb, self.wukong_mana_color_upperb)

    def _extract_wukong_stamina(self, observation):
        # Use image processing to extract the player's stamina from the observation
        # Placeholder; adjust this to use the actual player stamina bar region  
        return self._pixel_count(observation, self.wukong_stamina_bar_region,
                                 self.wukong_stamina_color_lowerb, self.wukong_stamina_color_upperb)
    
    
    
    def _pixel_count(self, img, area, 
                    lowerb, # lowerb Color RGB, Following pyautogui.screenshot
                    upperb, # lowerb Color RGB, Following pyautogui.screenshot
                    threshold=180):
        pixel_mask = cv2.inRange(img[area[1]:area[3], area[0]:area[2]], lowerb, upperb)
        return np.count_nonzero(pixel_mask.mean(axis=0) > threshold)
    
    def draw_areas(self, img): # assume img is RBG
        cv2.rectangle(img, self.wukong_health_bar_region[0:2], self.wukong_health_bar_region[2:4], (0, 255, 0), 2)
        cv2.rectangle(img, self.wukong_mana_bar_region[0:2], self.wukong_mana_bar_region[2:4], (0, 0, 255), 2)
        cv2.rectangle(img, self.wukong_stamina_bar_region[0:2], self.wukong_stamina_bar_region[2:4], (255, 255, 0), 2)
        cv2.rectangle(img, self.boss_health_bar_region[0:2], self.boss_health_bar_region[2:4], (255, 0, 0), 2)
        return img

    

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
import time

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
t1 = time.time()
frame_count = 0
for episode in range(episodes):
    state = env.reset()
    state = torch.FloatTensor(state).to(device)
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        if(not info['pasued']):
            # not pased
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(batch_size)
        # Display real-time window  r
        if(frame_count%20 == 0):
            print(next_state.shape)
            img = next_state 
            print(img[0][0])
            env.draw_areas(img)
            print(info)
            cv2.imshow('Game Analysis', cv2.cvtColor(cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        frame_count += 1
        if time.time() - t1 > 5:
            print('FPS:', frame_count / (time.time() - t1))
            t1 = time.time()
            frame_count = 0
        

    print(f"Episode: {episode}, Total Reward: {total_reward}")

cv2.destroyAllWindows()