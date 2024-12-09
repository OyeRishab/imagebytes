import torch
import torch.nn as nn

# Generator using DCGAN-like architecture
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bn=True):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(1, 64, bn=False),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.ConvTranspose2d(512, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# Discriminator using PatchGAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bn=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(4, 64, bn=False),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
