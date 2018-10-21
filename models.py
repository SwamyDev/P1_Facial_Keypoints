import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as init


def _init_conv(layer):
    return layer


def _init_dense(layer):
    init.xavier_uniform_(layer.weight)
    return layer


def _activation(x):
    return functional.elu(x)


def _flatten(x):
    return x.view(x.size(0), -1)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = _init_conv(nn.Conv2d(1, 32, 3))
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = _init_conv(nn.Conv2d(32, 64, 2))
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = _init_conv(nn.Conv2d(64, 128, 2))
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dens4 = _init_dense(nn.Linear(15488, 500))

        self.output = _init_dense(nn.Linear(500, 136))

    def forward(self, x):
        x = _activation(self.conv1(x))
        x = self.pool1(x)

        x = _activation(self.conv2(x))
        x = self.pool2(x)

        x = _activation(self.conv3(x))
        x = self.pool3(x)

        x = _flatten(x)

        x = _activation(self.dens4(x))

        return self.output(x)


class NaimishNet(nn.Module):
    def __init__(self):
        super(NaimishNet, self).__init__()
        self.conv1 = _init_conv(nn.Conv2d(1, 32, 4))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = _init_conv(nn.Conv2d(32, 64, 3))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = _init_conv(nn.Conv2d(64, 128, 2))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.3)

        self.conv4 = _init_conv(nn.Conv2d(128, 256, 1))
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(0.4)

        self.dens5 = _init_dense(nn.Linear(256 * 5 * 5, 1000))

        self.drop5 = nn.Dropout(0.5)

        self.dens6 = _init_dense(nn.Linear(1000, 1000))
        self.drop6 = nn.Dropout(0.6)

        self.output = _init_dense(nn.Linear(1000, 136))

    def forward(self, x):
        x = _activation(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = _activation(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = _activation(self.conv3(x))
        x = self.pool3(x)
        x = self.drop3(x)

        x = _activation(self.conv4(x))
        x = self.pool4(x)
        x = self.drop4(x)

        x = _flatten(x)

        x = _activation(self.dens5(x))
        x = self.drop5(x)

        x = self.dens6(x)
        x = self.drop6(x)

        x = self.output(x)

        return x
