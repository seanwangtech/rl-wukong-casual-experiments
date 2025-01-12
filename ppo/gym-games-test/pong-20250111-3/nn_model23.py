
# from efficientnet_pytorch import EfficientNet
import timm
import torch.nn as nn
import torch.nn
import numpy as np

class PPOnn(nn.Module):
    def __init__(self, num_actions):
        super(PPOnn, self).__init__()
        # self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        # self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        
        # # Remove the last fully connected layer by slicing (EfficientNet's output features are 1280)
        # self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.features = nn.Sequential(
            self.layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        # Custom fully connected layers for RL
        self.policy_fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # Custom fully connected layers for RL
        self.value_fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def layer_init(self,layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def forward(self, x):
        x = self.features(x)
        policy_logits = self.policy_fc(x)
        value = self.value_fc(x)
        return policy_logits, value
    