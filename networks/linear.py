from torch import nn


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim=256, num_classes=2):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
