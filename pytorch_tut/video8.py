# Video 8 - model analysis
# https://www.youtube.com/watch?v=UuteCccDXCE&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=8&ab_channel=sentdex

# dataset: https://www.microsoft.com/en-us/download/details.aspx?id=54765

import os
import cv2
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import time


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        # just runs once to get the size...
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:  # to iterate over dirs
            print(label)
            for f in tqdm(os.listdir(label)):  # to iterate over images in dirs
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # convert to one hot vector
                    self.training_data.append(
                        [np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    # print(str(e))
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)


# =================================================================
# main below - not in main function to make it more like the tut

REBUILD_DATA = False


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)

print(f"len(training_data): {len(training_data)}")

# net = Net().to(device)  # send to GPU


X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1  # val percent (validate?)
val_size = int(len(X)*VAL_PCT)
print(f"val_size): {val_size}")

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(f"len(train_X): {len(train_X)}")
print(f"len(test_X): {len(test_X)}")


# loss_function = nn.MSELoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)


def train(net):
    BATCH_SIZE = 100
    EPOCHS = 1

    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            # print(i, i+BATCH_SIZE)
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i+BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 28:57 for explanation (video 6)
            # do fitment, need to zero gradient
            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss}")


# train(net)


def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            # net_out already on the device
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy:", round(correct/total, 3))


# test(net)


def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss


def test(size=32):
    X, y = test_X[:size], test_y[:size]
    random_start = np.random.randint(len(test_X)-size)
    X = test_X[random_start:random_start + size]
    y = test_y[random_start:random_start + size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(
            X.view(-1, 1, 50, 50).to(device), y.to(device))
    return val_acc, val_loss


# val_acc, val_loss = test()
# print(val_acc, val_loss)


MODEL_NAME = f"model-{int(time.time())}"
print(MODEL_NAME)

net = Net().to(device)  # send to GPU
loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


def train():
    BATCH_SIZE = 100
    EPOCHS = 8
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i +
                                  BATCH_SIZE].view(-1, 1, 50, 50).to(device)
                batch_y = train_y[i:i+BATCH_SIZE].to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)
                if i % 50 == 0:  # every 50 steps
                    val_acc, val_loss = test(size=100)
                    f.write(
                        f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")


train()
