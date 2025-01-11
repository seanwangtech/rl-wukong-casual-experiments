
# from efficientnet_pytorch import EfficientNet
import timm
import torch.nn as nn
import gymnasium as gym
import numpy as np
import torch as th
# idea from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
class DQNEfficientNet(nn.Module):
    def __init__(self, num_actions, 
                 n_input_channels = 3,
                 observation_space=gym.spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)):
        super(DQNEfficientNet, self).__init__()
        # self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        # self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        
        # # Remove the last fully connected layer by slicing (EfficientNet's output features are 1280)
        # self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.features = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.features(th.as_tensor(observation_space.sample().transpose(2, 0, 1)[None]).float()).shape[1]
        # Custom fully connected layers for RL
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
    