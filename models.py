## TODO: define the convolutional neural network architecture

import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = self._init_conv(nn.Conv2d(1, 32, 3))
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = self._init_conv(nn.Conv2d(32, 64, 2))
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = self._init_conv(nn.Conv2d(64, 128, 2))
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dens4 = self._init_dense(nn.Linear(15488, 500))

        self.output = self._init_dense(nn.Linear(500, 136))

    @staticmethod
    def _init_conv(layer):
        return layer

    @staticmethod
    def _init_dense(layer):
        I.xavier_uniform_(layer.weight)
        return layer

    def forward(self, x):
        x = self._activation(self.conv1(x))
        x = self.pool1(x)

        x = self._activation(self.conv2(x))
        x = self.pool2(x)

        x = self._activation(self.conv3(x))
        x = self.pool3(x)

        x = self._flatten(x)

        x = self._activation(self.dens4(x))

        return self.output(x)

    @staticmethod
    def _activation(x):
        return F.elu(x)

    @staticmethod
    def _flatten(x):
        return x.view(x.size(0), -1)
