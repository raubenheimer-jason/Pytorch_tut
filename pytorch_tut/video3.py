# Video 3
# https://www.youtube.com/watch?v=ixathu7U-LQ&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=3&ab_channel=sentdex

from random import shuffle
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)  # 28*28 = 784, the image size
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def testing():
    net = Net()

    X = torch.rand((28, 28))
    X = X.view(-1, 28*28)
    output = net(X)
    print(output)


def main():
    train = datasets.MNIST("", train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))

    test = datasets.MNIST("", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


if __name__ == "__main__":
    # main()
    testing()
