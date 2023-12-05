import torch
from torchvision.models import resnet18
from torch import nn

# Initialize the ResNet18 Model
# Device agnostic code:
device = torch.device("cpu")
model = resnet18()

# Define the modified fully connected layer:
class resnetfc(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features= 8)
        )
    def forward(self, x):
        return self.classifier(x)

# Assign the new fc layer to the intitialized model
model.fc = resnetfc()
# Load the saved "State Dictionary" of the trained model in Jupyter Notebook
model = model.load_state_dict(torch.load("FacialEmo.pth", map_location=device))