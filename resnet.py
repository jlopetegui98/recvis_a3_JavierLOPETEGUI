import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nclasses = 500  # Number of output classes


class ResnetClf(nn.Module):
    def __init__(self):
        super(ResnetClf, self).__init__()
        # Load a pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=True)
        # keep resnet features frozen
        for param in self.resnet.parameters():
            param.requires_grad = False
        # add a linear layer on top
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, nclasses)

    def forward(self, x):
        # Forward pass through ResNet
        return self.resnet(x)