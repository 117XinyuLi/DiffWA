import torch.nn as nn

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNRelu, self).__init__()

        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, stride, padding=1)
        self.bn2 = nn.BatchNorm2d(channels_out)

        self.conv3 = nn.Sequential()
        if channels_in != channels_out:
            self.conv3 = nn.Conv2d(channels_in, channels_out, 1, stride, padding=0)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x + self.conv3(res)
