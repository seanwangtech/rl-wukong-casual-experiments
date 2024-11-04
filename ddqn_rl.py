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
                 wukong_health_color_lowerb = (80, 70, 70),  # Color is RGB
                 wukong_health_color_upperb = (230, 230, 230),
                 wukong_mana_color_lowerb = (45, 80, 130),
                 wukong_mana_color_upperb = (85, 140, 210),
                 wukong_stamina_color_lowerb = (110, 130, 75),
                 wukong_stamina_color_upperb = (200, 165, 110),
                 boss_health_color_lowerb = (170, 170, 170),
                 boss_health_color_upperb = (230, 230, 230),
                 ):
        super(BlackMythWukongEnv, self).__init__()
        self.action_space = spaces.Discrete(5)  # 0: dodge, 1: use_gourd, 2: light_attack, 3: heavy_attack, 4: do nothing
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
            # elif action == 1:
            #     pyautogui.press('r')           # Use Gourd
            elif action == 1:
                pyautogui.press('ctrl')           # jump
            elif action == 2:
                pyautogui.click(button='left')  # Light Attack
            elif action == 3:
                pyautogui.click(button='right') # Heavy Attack
            elif action == 4:
                pass # do nothing

        # Obtain new observation and calculate reward
        observation = self._get_observation()
        pixel_count_meta = {
            'wukong_health': self._extract_wukong_health(observation),
            'wukong_mana': self._extract_wukong_mana(observation),
            'wukong_stamina': self._extract_wukong_stamina(observation),
            'boss_health': self._extract_boss_health(observation)
        }
        reward = self._calculate_reward(pixel_count_meta)
        done = False  # Define condition for end of game if possible
        if(pixel_count_meta['wukong_health'] == 0
           and pixel_count_meta['wukong_stamina'] > 0):
            if not self.paused: 
                done = True
                # cv2.imwrite(f'game_over-{time.time()}.png', cv2.cvtColor(observation, cv2.COLOR_RGB2BGR))
            self.paused = True

        return observation, reward, done, {'pasued': self.paused,
                                            **pixel_count_meta}

    def _get_observation(self):        
        # Capture screenshot of the game window
        screenshot = pyautogui.screenshot(region=(0, 0, self.screen_width, self.screen_height))
        img = np.array(screenshot)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # no need to do covert, the pyautogui screenshot already in RGB
        # resized_img = cv2.resize(img, (224*2, 224*2))
        return img

    def _calculate_reward(self, meta):
        # check if the object has attribute '_previous_meta_for_reward'
        if not hasattr(self, '_previous_meta_for_reward'):
            self._previous_meta_for_reward = meta
            return 0
        # if(meta['wukong_health'] == 0):
        #     return -200
        # calculate reward
        previous_meta = self._previous_meta_for_reward
        self._previous_meta_for_reward = meta
        previous_boss_health = previous_meta['boss_health']
        boss_health = meta['boss_health']
        boss_health_reward = (previous_boss_health - boss_health) if boss_health < previous_boss_health else 0
        previous_wukong_health = previous_meta['wukong_health']
        wukong_health = meta['wukong_health']
        wukong_health_reward = (wukong_health - previous_wukong_health)*0.3 if wukong_health < previous_wukong_health else 0
        reward = boss_health_reward + wukong_health_reward
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
    def __init__(self, action_space, policy_net, target_net, optimizer, loss_fn, gamma=0.99, device='cpu'):
        self.action_space = action_space
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.memory = deque(maxlen=2000)
        self.policy_net = policy_net
        self.target_net = target_net  # Set target model for DDQN
        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()
        self.epsilon = 0.4  # Epsilon-greedy action selection
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.device = device

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def select_action(self, state):
        if random.random() <= self.epsilon:
            return self.action_space.sample() # Explore
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
        return q_values.argmax().item()   # Exploit

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        # Double DQN: Use policy net to choose action, target net to calculate Q-value of that action
        with torch.no_grad():
            # Get the best action from the policy network
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # Use target network to get Q-values for these actions
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # for state, action, reward, next_state, done in batch:
        #     target = reward
        #     if not done:
        #         next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        #         target += self.gamma * self.target_model(next_state).max(1)[0].item()
        #     target_f = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        #     target_f[0][action] = target
        #     loss = self.loss_fn(target_f, torch.FloatTensor(target).unsqueeze(0).to(self.device))
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
            
            
import torch.nn.functional as F
import torch.optim as optim
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = BlackMythWukongEnv()
policy_net = DQNEfficientNet(env.action_space.n)
target_net = DQNEfficientNet(env.action_space.n)
loss_fn = F.mse_loss
# Assuming `model` is an instance of DQNEfficientNet and learning_rate is defined
learning_rate = 1e-4

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam([
    {'params': policy_net.features.parameters(), 'lr': 1e-5},  # Low LR for pre-trained backbone
    {'params': policy_net.fc.parameters(), 'lr': 1e-3}  # Higher LR for classifier
],lr=0.001)
# for param in model.features.parameters():
#     param.requires_grad = False  # Freeze pre-trained layers

agent = DDQNAgent(env.action_space, 
                  policy_net=policy_net,
                  target_net=target_net,
                  optimizer=optimizer, loss_fn=loss_fn, device=device)

episodes = int(1e9)
batch_size = 4

def obs2stateTensor(obs, device, show=False):
    obs = obs[:, 420:1500] # choose area for model input
    state = cv2.resize(obs, (224, 224))
    if(show): 
        cv2.imshow('model input', cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
    state = torch.FloatTensor(state).to(device)
    state = state.permute(2, 0, 1)  # (H, W, C) -> C, H, W
    return state

t1 = time.time()
for episode in range(episodes):
    state = env.reset()
    state = obs2stateTensor(state, device, show=True)

    done = False
    total_reward = 0
    
    frame_count = 0
    while not done:
        action = agent.select_action(state)
        obs_img, reward, done, info = env.step(action)
        if(not info['pasued']):
            # not pased
            total_reward += reward
            # print('reward',reward)
            next_state = obs2stateTensor(obs_img, device, show=True)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(batch_size)
            agent.update_epsilon()
            if(frame_count%20==0):
                agent.update_target_net()
            total_reward += reward
        
        if time.time() - t1 > 1:
            FPS = frame_count / (time.time() - t1)
            frame_count = 0
            t1 = time.time()
            img = obs_img 
            env.draw_areas(img)
            print(f'[{"Paused" if info["pasued"] else "Trainning"}] Ep: {episode}, FPS: {FPS:.2f}, total reward: {total_reward:.2f}, \
Epsilon: {agent.epsilon:.2f}, \
wukong (H, M, S): {(info["wukong_health"], info["wukong_mana"], info["wukong_stamina"])}, boss: {info["boss_health"]}')
            cv2.imshow('Game Analysis', cv2.cvtColor(cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(5):
                break
        frame_count += 1
        time.sleep(0.005)
    time.sleep(0.2)
    if(frame_count>5):
        print(f"Episode Done: {episode}, Total Reward: {total_reward}")

cv2.destroyAllWindows()