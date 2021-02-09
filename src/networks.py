"""

"""
import torch
import torch.nn as nn


class Conv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation,
            normalization):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.normalization = normalization(out_channels)
        self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.normalization(x)
        return self.activation(x)


class ResBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size):

        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.PReLU,
            normalization=nn.BatchNorm2d)

        self.conv2 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.Identity,
            normalization=nn.BatchNorm2d)

    def _residual(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        return x + self._residual(x)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=9,
            stride=1,
            padding=9//2,
            activation=nn.PReLU,
            normalization=nn.BatchNorm2d)

    def forward(self, x):
        x = self.conv1(x)
        return x


generator = Generator()

x = torch.rand((1, 3, 256, 256), dtype=torch.float32)

print(x.shape)
x = generator(x)
print(x.shape)
