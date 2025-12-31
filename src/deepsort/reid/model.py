# src/deepsort/reid/model.py
import torch.nn as nn
from torchvision.models import resnet50

class Net(nn.Module):
    def __init__(self, feature_dim=128):
        super(Net, self).__init__()
        self.base = resnet50(pretrained=False)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, feature_dim)

    def forward(self, x):
        feat = self.base(x)
        return nn.functional.normalize(feat, p=2, dim=1)
