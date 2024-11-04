
import gymnasium as gym
import cv2
import numpy as np
import pyautogui
from gymnasium import spaces
from pynput import keyboard
import time


class BlackMythWukongEnv(gym.Env):
    def __init__(self,
                 wukong_health_bar_region=(208, 982, 500, 992), # (x1, y1, x2, y2), for 1920x1080, the area is (208, 982, 600, 992)
                 wukong_mana_bar_region=(208, 1002, 600, 1008), # (x1, y1, x2, y2), for 1920x1080, the area is (208, 1002, 600, 1008)
                 wukong_stamina_bar_region=(208, 1016, 600, 1021), # (x1, y1, x2, y2), for 1920x1080, the area is (208, 1016, 600, 1021)
                 wukong_focus_bar_region=(1695, 952, 1740, 1040), # (x1, y1, x2, y2), for 1920x1080, the area is (1695, 952, 1740, 1040)
                 wukong_focus_point_region = (1780, 985, 1835, 1050), # (x1, y1, x2, y2), for 1920x1080, the area is (1780, 985, 1835, 1050)
                 boss_health_bar_region=(760, 912, 1172, 922), # (x1, y1, x2, y2), for 1920x1080, the area is (760, 912, 1172, 922)
                 wukong_health_color_lowerb = (50, 50, 50),  # Color is RGB
                 wukong_health_color_upperb = (255, 255, 255),
                 wukong_mana_color_lowerb = (45, 80, 130),
                 wukong_mana_color_upperb = (85, 140, 210),
                 wukong_stamina_color_lowerb = (110, 130, 75),
                 wukong_stamina_color_upperb = (200, 165, 110),
                 wukong_focus_color_lowerb = (180, 180, 180),
                 wukong_focus_color_upperb = (255, 255, 255),
                 wukong_focus_point_color_lowerb = (249, 220, 100),
                 wukong_focus_point_color_upperb = (255, 255, 255),
                 boss_health_color_lowerb = (170, 170, 170),
                 boss_health_color_upperb = (250, 230, 230),
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
        self.wukong_focus_bar_region = wukong_focus_bar_region
        self.wukong_focus_point_region = wukong_focus_point_region
        self.boss_health_bar_region = boss_health_bar_region

        self.wukong_health_color_lowerb = wukong_health_color_lowerb
        self.wukong_health_color_upperb = wukong_health_color_upperb
        self.wukong_mana_color_lowerb = wukong_mana_color_lowerb
        self.wukong_mana_color_upperb = wukong_mana_color_upperb
        self.wukong_stamina_color_lowerb = wukong_stamina_color_lowerb
        self.wukong_stamina_color_upperb = wukong_stamina_color_upperb
        self.wukong_focus_color_lowerb = wukong_focus_color_lowerb
        self.wukong_focus_color_upperb = wukong_focus_color_upperb
        self.wukong_focus_point_color_lowerb = wukong_focus_point_color_lowerb
        self.wukong_focus_point_color_upperb = wukong_focus_point_color_upperb
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
                time.sleep(0.2)
            # elif action == 1:
            #     pyautogui.press('r')           # Use Gourd
            elif action == 1:
                pyautogui.click(button='left')         # light attach
                time.sleep(0.5)
                pyautogui.click(button='right')        # heavy attack
                time.sleep(1.0)
            elif action == 2:
                pyautogui.click(button='left')  # Light Attack
                time.sleep(0.5)
            elif action == 3:
                pyautogui.click(button='right') # Heavy Attack
                time.sleep(1.2)
            elif action == 4:
                time.sleep(0.1) # do nothing

        # Obtain new observation and calculate reward
        observation = self._get_observation()
        pixel_count_meta = {
            'wukong_health': self._extract_wukong_health(observation),
            'wukong_mana': self._extract_wukong_mana(observation),
            'wukong_stamina': self._extract_wukong_stamina(observation),
            'wukong_focus_bar': self._extract_wukong_focus_bar(observation),
            'wukong_focus_point': self._extract_wukong_focus_point(observation),
            'boss_health': self._extract_boss_health(observation)
        }
        pixel_count_meta['wukong_focus'] = self._calc_wukong_focus(pixel_count_meta['wukong_focus_bar'], pixel_count_meta['wukong_focus_point']) 
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
        if(meta['wukong_health'] == 0):
            return 0
        # if(meta['wukong_health'] == 0):
        #     return -200
        # calculate reward
        step_reward = 0.1
        previous_meta = self._previous_meta_for_reward
        self._previous_meta_for_reward = meta
        previous_boss_health = previous_meta['boss_health']
        boss_health = meta['boss_health']
        boss_health_reward = (previous_boss_health - boss_health) if boss_health - previous_boss_health < -2 else 0
        previous_wukong_health = previous_meta['wukong_health']
        wukong_health = meta['wukong_health']
        wukong_health_reward = (wukong_health - previous_wukong_health) if wukong_health - previous_wukong_health <-2 else 0
        reward = boss_health_reward + wukong_health_reward + step_reward
        return reward

    def _extract_boss_health(self, observation):
        # Use image processing to extract the boss's health from the observation
        # Placeholder; adjust this to use the actual boss health bar region
        return self._pixel_count_width(observation, self.boss_health_bar_region,
                                 self.boss_health_color_lowerb, self.boss_health_color_upperb)

    def _extract_wukong_health(self, observation):
        # Use image processing to extract the player's health from the observation
        # Placeholder; adjust this to use the actual player health bar region
        return self._pixel_count_width(observation, self.wukong_health_bar_region,
                                 self.wukong_health_color_lowerb, self.wukong_health_color_upperb)
        
    def _extract_wukong_mana(self, observation):
        # Use image processing to extract the player's mana from the observation
        # Placeholder; adjust this to use the actual player mana bar region
        return self._pixel_count_width(observation, self.wukong_mana_bar_region,
                                 self.wukong_mana_color_lowerb, self.wukong_mana_color_upperb)

    def _extract_wukong_stamina(self, observation):
        # Use image processing to extract the player's stamina from the observation
        # Placeholder; adjust this to use the actual player stamina bar region  
        return self._pixel_count_width(observation, self.wukong_stamina_bar_region,
                                 self.wukong_stamina_color_lowerb, self.wukong_stamina_color_upperb)

    
    def _extract_wukong_focus_bar(self, observation):
        # Use image processing to extract the player's focus from the observation
        # Placeholder; adjust this to use the actual player focus bar region
        region_width = abs(self.wukong_focus_bar_region[2] - self.wukong_focus_bar_region[0])
        return self._pixel_count_height(observation, self.wukong_focus_bar_region,
                                 self.wukong_focus_color_lowerb, self.wukong_focus_color_upperb, threshold=3/region_width*255)
    def _extract_wukong_focus_point(self, observation):
        # Use image processing to extract the player's focus from the observation
        # Placeholder; adjust this to use the actual player focus bar region
        region_width = abs(self.wukong_focus_point_region[2] - self.wukong_focus_point_region[0])
        count = self._pixel_count_height(observation, self.wukong_focus_point_region,
                                 self.wukong_focus_point_color_lowerb, self.wukong_focus_point_color_upperb, threshold= 5/region_width*255)
        if(count <5):
            return 0
        elif(count <16):
            return 1
        elif(count <30):
            return 2
        else:
            return 3
    def _calc_wukong_focus(self, focusBar, focusPoint):
        focusBarHeight = abs(self.wukong_focus_bar_region[3] - self.wukong_focus_bar_region[1])
        return focusBar + focusPoint * focusBarHeight
    def _pixel_count_width(self, img, area, 
                    lowerb, # lowerb Color RGB, Following pyautogui.screenshot
                    upperb, # lowerb Color RGB, Following pyautogui.screenshot
                    threshold=180):
        pixel_mask = cv2.inRange(img[area[1]:area[3], area[0]:area[2]], lowerb, upperb)
        return np.count_nonzero(pixel_mask.mean(axis=0) > threshold)
    
    def _pixel_count_height(self, img, area, 
                    lowerb, # lowerb Color RGB, Following pyautogui.screenshot
                    upperb, # lowerb Color RGB, Following pyautogui.screenshot
                    threshold=180):
        pixel_mask = cv2.inRange(img[area[1]:area[3], area[0]:area[2]], lowerb, upperb)       
        return np.count_nonzero(pixel_mask.mean(axis=1) > threshold)
    
    def draw_areas(self, img): # assume img is RBG
        cv2.rectangle(img, self.wukong_health_bar_region[0:2], self.wukong_health_bar_region[2:4], (0, 255, 0), 2)
        cv2.rectangle(img, self.wukong_mana_bar_region[0:2], self.wukong_mana_bar_region[2:4], (0, 0, 255), 2)
        cv2.rectangle(img, self.wukong_stamina_bar_region[0:2], self.wukong_stamina_bar_region[2:4], (255, 255, 0), 2)
        cv2.rectangle(img, self.wukong_focus_bar_region[0:2], self.wukong_focus_bar_region[2:4], (0, 255, 255), 2)
        cv2.rectangle(img, self.wukong_focus_point_region[0:2], self.wukong_focus_point_region[2:4], (255, 0, 255), 2)
        cv2.rectangle(img, self.boss_health_bar_region[0:2], self.boss_health_bar_region[2:4], (255, 0, 0), 2)
        return img
