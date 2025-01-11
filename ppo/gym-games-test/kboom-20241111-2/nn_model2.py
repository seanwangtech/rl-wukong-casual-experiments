
import timm
import torch.nn as nn

class PPOnn(nn.Module):
    def __init__(self, num_actions):
        super(PPOnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.features_n = 2464
        # Custom fully connected layers for RL
        self.policy_fc = nn.Sequential(
            nn.Linear(self.features_n, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
                # Custom fully connected layers for RL
        self.value_fc = nn.Sequential(
            nn.Linear(self.features_n, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self, x):
        x = self.features(x)
        policy_logits = self.policy_fc(x)
        value = self.value_fc(x)
        return policy_logits, value
    