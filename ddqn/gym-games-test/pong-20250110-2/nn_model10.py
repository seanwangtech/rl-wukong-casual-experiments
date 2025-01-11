
# from efficientnet_pytorch import EfficientNet
import timm
import torch.nn as nn

class DQNEfficientNet(nn.Module):
    def __init__(self, num_actions):
        super(DQNEfficientNet, self).__init__()
        # self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        # self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        
        # # Remove the last fully connected layer by slicing (EfficientNet's output features are 1280)
        # self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=10, stride=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # Custom fully connected layers for RL
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
    