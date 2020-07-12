#  Author : LUKOMBO Christopher
#  Version : 17

import tensorflow as tf
from predict.Constantes import Constantes
import os
import random
import base64
import cv2
import matplotlib.pyplot as plt

class Predict:

    def __read_picture(self, file):
        img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (Constantes.IMG_SIZE, Constantes.IMG_SIZE))
        return new_array.reshape(-1, Constantes.IMG_SIZE, Constantes.IMG_SIZE, 1)

    def __picture_to_base64(self, file):
        picture = None
        with open(file, "rb") as img_file:
            picture = base64.b64encode(img_file.read())
        return picture

    def predict_display(self, classes):
        model = tf.keras.models.load_model("CNN.model")
        image = "8587.jpg"
        picture_array = self.__read_picture(image)
        prediction = model.predict([picture_array])
        prediction = list(prediction[0])

        prediction_ = classes[prediction.index(max(prediction))]

    def predict(self, labels, name):
        model = tf.keras.models.load_model("CNN.model")

        fig, ax = plt.subplots(Constantes.ROWS, Constantes.COLS, frameon=False, figsize=(15, 25))
        fig.suptitle('Random Image from Each Food Class', fontsize=20)
        sorted_food_dirs = sorted(os.listdir(Constantes.ROOT_DIRECTORY))

        image = None
        imageFound  = []
        for i in range(Constantes.ROWS):
            for j in range(Constantes.COLS):
                try:
                    food_directory = sorted_food_dirs[i * Constantes.COLS + j]
                except:
                    break
                all_files = os.listdir(os.path.join(Constantes.ROOT_DIRECTORY, food_directory))
                for k in range(len(all_files)):
                    if k == Constantes.NB_IMAGES2:
                        break
                    k = k + 1
                    image = all_files[random.randint(0, len(all_files)-1)]

                    path = Constantes.ROOT_DIRECTORY + food_directory

                    new_array = self.__read_picture(os.path.join(path, image))

                    picture_array = new_array
                    prediction = model.predict([picture_array])
                    prediction = list(prediction[0])

                    prediction_ = labels[prediction.index(max(prediction))]
                    if prediction_.lower().find(name.lower()) >= 0:

                        #Leaving print on purpose for presentation
                        print(food_directory, ' : ', prediction_)
                        print()
                        imageFound.append(self.__picture_to_base64(os.path.join(path, image)))



        if imageFound is None:
            imageFound = ''

        return imageFound