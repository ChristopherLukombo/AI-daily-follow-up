#  Author : LUKOMBO Christopher, DELIESSCHE Angelo
#  Version : 17

import pickle

import matplotlib.pyplot as plt
from tensorflow.python.keras import Sequential, regularizers
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense


class Train:

    def load_X_and_Y(self):
        X = pickle.load(open("X.pickle", "rb"))
        # normalizing data for pixel (from 0 to 255)
        X = X / 255.0
        y = pickle.load(open("y.pickle", "rb"))
        return X, y

    def build_model(self, X):
        model = Sequential()

        # 3 convolutional layers
        model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 2 hidden layers
        model.add(Flatten())
        model.add(Dense(258, kernel_regularizer=regularizers.l2(0.001)))
        model.add(Activation("relu"))
        #model.add(Dropout(0.3))


        model.add(Dense(258, kernel_regularizer=regularizers.l2(0.001)))
        model.add(Activation("relu"))
        #model.add(Dropout(0.5))
        model.add(Dense(258))
        model.add(Activation("relu"))



        # The output layer with 120 neurons, for 120 classes
        model.add(Dense(10))
        model.add(Activation("softmax"))
        return model

    def compile_model(self, model):
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

    def train_model(self, model, X, y):

        return model.fit(X, y, batch_size=128, epochs=400, validation_split=0.2)

    def save_model(self, model):
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")

        print("Saved model")
        model.save('CNN.model')

    def display_infos(self, result):
        print(result.history.keys())
        plt.figure(1)
        plt.plot(result.history['accuracy'])
        plt.plot(result.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()