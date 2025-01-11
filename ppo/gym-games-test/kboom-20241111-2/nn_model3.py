import timm
import torch
import torch.nn as nn

class PPOnn(nn.Module):
    def __init__(self, num_actions):
        super(PPOnn, self).__init__()
        
        # Load a lightweight pre-trained ResNet-18 without the fully connected layer
        self.feature_extractor = timm.create_model('resnet18', pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])  # Remove the last FC layer
        self.feature_dim = 512  # ResNet-18 output features after pooling

        # Policy network head
        self.policy_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # Value network head
        self.value_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Extract features using ResNet backbone
        x = self.feature_extractor(x).view(x.size(0), -1)  # Flatten the output
        
        # Compute policy and value outputs
        policy_logits = self.policy_fc(x)
        value = self.value_fc(x)
        
        return policy_logits, value