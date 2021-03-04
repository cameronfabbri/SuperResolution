"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)

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

        # initialize our weights to be very low since this is a very deep network
        self.conv.apply(init_weights)

        if normalization is not None:
            self.normalization = normalization(out_channels)
        else:
            self.normalization = nn.Identity()
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
        super(ResBlock, self).__init__()

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

# as defined by the ESRGAN paper
# TODO: make the lnumber of residuable blocks configurable
class ResBlockDense(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size):
        super(ResBlockDense, self).__init__()

        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.LeakyReLU,
            normalization=None)
        self.conv2 = Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.LeakyReLU,
            normalization=None)
        self.conv3 = Conv2d(
            in_channels=in_channels * 3,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.LeakyReLU,
            normalization=None)
        self.conv4 = Conv2d(
            in_channels=in_channels * 4,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.LeakyReLU,
            normalization=None)
        # The last summation convolution before we leave the resblock
        self.conv5 = Conv2d(
            in_channels=in_channels * 4,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.Identity,
            normalization=None)

    #def _residual(self,x):
    #    skip1 = x
    #    x = self.conv1(x) + skip1
    #    skip2 = x
    #    x = self.conv2(x) + skip1 + skip2
    #    skip3 = x
    #    x = self.conv3(x) + skip1 + skip2 + skip3
    #    #x = self.conv4(x) + skip1 + skip2 + skip3
    #    x = self.conv4(x)
    #    x = torch.cat([x, skip1, skip2, skip3], 1)
    #    x = self.conv5(x)
    #    return x

    def _residual(self, x):
        skip = x
        c1 = self.conv1(x)
        c2 = self.conv2(torch.cat([skip, c1], 1))
        c3 = self.conv3(torch.cat([skip, c1, c2], 1))
        c4 = self.conv4(torch.cat([skip, c1, c2, c3], 1))
        c5 = self.conv5(torch.cat([skip, c1, c2, c4], 1))

        return c5

    def forward(self, x):
        # sum our entry with the result of our residual
        # not current using the residual scaling factor β
        return x + self._residual(x)



class SISR_Resblocks(nn.Module):
    def __init__(self, num_blocks):
        super(SISR_Resblocks, self).__init__()

        self.resblocks = []
        for i in range(num_blocks):
            self.resblocks.append(
                ResBlock(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3))
        self.resblocks = nn.Sequential(*self.resblocks)

    def forward(self, x):
        return self.resblocks(x)


class Generator(nn.Module):

    def __init__(self, resblocks):
        super(Generator, self).__init__()

        self.conv1 = Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=9,
            stride=1,
            padding=9//2,
            activation=nn.PReLU,
            normalization=None)

        self.resblocks = []
        for i in range(10):
            self.resblocks.append(
                #ResBlock(
                #    in_channels=64,
                #    out_channels=64,
                #    kernel_size=3)
                ResBlockDense(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3)
                )
        self.resblocks = nn.Sequential(*self.resblocks)

        # ESRGAN poo-poos batch norm layers
        #self.conv2 = Conv2d(
        #    in_channels=64,
        #    out_channels=64,
        #    kernel_size=3,
        #    stride=1,
        #    padding=1,
        #    activation=nn.Identity,
        #    normalization=nn.BatchNorm2d)

        self.resblocks = resblocks

        self.conv3 = Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.Identity,
            normalization=None)

        # this is effectively our upscale factor
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.prelu = nn.PReLU()

        self.conv4 = Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=9,
            stride=1,
            padding=9//2,
            activation=nn.Tanh,
            normalization=None)

    def forward(self, x):
        skip = self.conv1(x)
        # ignore the residual scaling paramter β for now
        # add the skip back at the end of our resblocks
        x = self.resblocks(skip) + skip
        #x = self.conv2(x) + skip
        #x = self.conv3(x)
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        x = self.conv4(x)
        return x