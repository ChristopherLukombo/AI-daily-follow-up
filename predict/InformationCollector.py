#  Author : LUKOMBO Christopher
#  Version : 17


import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from predict.Constantes import Constantes

fig, ax = plt.subplots(Constantes.ROWS, Constantes.COLS, frameon=False, figsize=(15, 25))
fig.suptitle('Random Image from Each Food Class', fontsize=20)
sorted_food_dirs = sorted(os.listdir(Constantes.ROOT_DIRECTORY))

class InformationCollector:
    def __init__(self):
        self.classes = self.__load_classes(Constantes.CLASS_PATH)
        self.labels = self.__load_labels(Constantes.LABELS_PATH)

    def __load_classes(self, class_path):
        classes = []
        with open(class_path) as data_file:
            for line in data_file:
                if len(line) == 0:
                    continue
                classes.append(line.rstrip())
        return classes

    def __load_labels(self, labels_path):
        labels = []
        with open(labels_path, encoding="utf-8") as data_file:
            for line in data_file:
                if len(line) == 0:
                    continue
                labels.append(line.rstrip())
        return labels

    def get_classes(self):
        return self.classes

    def get_labels(self):
        return self.labels

    def create_training_data(self):
        training_data = []
        for i in range(Constantes.ROWS):
            for j in range(Constantes.COLS):
                try:
                    food_directory = sorted_food_dirs[i * Constantes.COLS + j]
                except:
                    break
                all_files = os.listdir(os.path.join(Constantes.ROOT_DIRECTORY, food_directory))
                for k in range(len(all_files)):
                    if k == Constantes.NB_IMAGES:
                        break
                    k = k + 1
                    image = all_files[k]

                    path = Constantes.ROOT_DIRECTORY + food_directory

                    class_num = self.classes.index(food_directory)

                    img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                    # Resize of pictures
                    new_array = cv2.resize(img_array, (Constantes.IMG_SIZE, Constantes.IMG_SIZE))

                    training_data.append([new_array, class_num])

        return training_data

    def mix(self, training_data):
        random.shuffle(training_data)

    def fill_X_and_y(self, training_data):
        X = []  # features
        y = []
        for features, label in training_data:
            X.append(features)
            y.append(label)

        y = np.array(y)
        return X, y

    def reshape(self, X):

        return np.array(X).reshape(-1, Constantes.IMG_SIZE, Constantes.IMG_SIZE, 1)

    def save_infos(self, X, y):
        pickle_out = open("X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()
        pickle_out = open("y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
        pickle_in = open("X.pickle", "rb")
        X = pickle.load(pickle_in)
