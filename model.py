import torch
import torch.nn as nn

class ColorizationUNet(nn.Module):
    def __init__(self):
        super(ColorizationUNet, self).__init__()

        def block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )

        self.encoder = nn.Sequential(
            block(1, 64),
            block(64, 128),
            block(128, 256)
        )

        self.decoder = nn.Sequential(
            block(256, 128),
            block(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Outputs normalized values
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
