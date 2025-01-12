
# from efficientnet_pytorch import EfficientNet
import timm
import torch.nn as nn

class PPOnn(nn.Module):
    def __init__(self, num_actions):
        super(PPOnn, self).__init__()
        # self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        # self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        
        # # Remove the last fully connected layer by slicing (EfficientNet's output features are 1280)
        # self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        # Custom fully connected layers for RL
        self.policy_fc = nn.Sequential(
            nn.Linear(512, num_actions)
        )
        
                # Custom fully connected layers for RL
        self.value_fc = nn.Sequential(
            nn.Linear(512, 1)
        )


    def forward(self, x):
        x = self.features(x)
        policy_logits = self.policy_fc(x)
        value = self.value_fc(x)
        return policy_logits, value
    