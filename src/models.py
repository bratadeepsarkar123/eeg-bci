import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """Compact CNN for EEG classification (Lawhern et al., 2018)."""
    def __init__(self, n_chan=16, n_time=32):
        super(EEGNet, self).__init__()
        # Block 1: Temporal & Spatial Convolutions
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 8, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (n_chan, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25)
        )
        # Block 2: Separable Convolutions
        self.b2 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 8), groups=16, padding='same', bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25)
        )
        self.fc = nn.LazyLinear(2)

    def forward(self, x):
        return self.fc(self.b2(self.b1(x)).view(x.size(0), -1))
