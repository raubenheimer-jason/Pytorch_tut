# Video 5 - convnet
# https://www.youtube.com/watch?v=9aYuQmMJvjA&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=5&ab_channel=sentdex

# dataset: https://www.microsoft.com/en-us/download/details.aspx?id=54765

import os
import cv2
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

REBUILD_DATA = False


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


def main():
    # pass

    if REBUILD_DATA:
        dogsvcats = DogsVSCats()
        dogsvcats.make_training_data()

    training_data = np.load("training_data.npy", allow_pickle=True)

    print(f"len(training_data): {len(training_data)}")

    print(training_data[0])

    plt.imshow(training_data[0][0], cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
