import torch
from torchvision.models import resnet18
from torch import nn

# Initialize the ResNet18 Model
# Device agnostic code:
DEVICE = torch.device("cpu")
MODEL = resnet18()

# Define the modified fully connected layer:
class resnetfc(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features= 8)
        )
    def forward(self, x):
        return self.classifier(x)
