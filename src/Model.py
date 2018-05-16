import keras
from keras.datasets import mnist
import numpy as np
from PIL import Image
from src.Pipeline import Pipeline

#import pydot

class Model():
    def __init__(self):
        self.model = None
        self.augmented_test_images = None
        self.augmented_train_images = None
        self.num_classes = 14
        self.batch_size = 256
        self.epochs = 3

    def gen_data(self):
        print("Start generating augmented images")
        p = Pipeline("./synthetic")
        p.random_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=1)
        p.random_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=1)
        p.random_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=1)
        self.augmented_train_images = p.sample(6000)
        self.augmented_test_images = p.sample(1000)

    def train(self):
        """LOAD MNIST DATABASE"""
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)

        """WORKING WITH TRAIN DATA"""
        for i, k in enumerate(self.augmented_train_images):
            x_train = np.concatenate((x_train, k), axis=0)
            y_train = np.append(y_train, [i + 10] * len(k))

        perm = np.random.permutation(6000*self.num_classes)
        x_temp = x_train.copy()
        y_temp = y_train.copy()
        for i in range(len(perm)):
            x_train[i] = x_temp[perm[i]]
            y_train[i] = y_temp[perm[i]]

        x_train = x_train.reshape(6000*self.num_classes, 28, 28, 1)

        """WORKING WITH TEST DATA"""
        for i, k in enumerate(self.augmented_test_images):
            x_test = np.concatenate((x_test, k), axis=0)
            y_test = np.append(y_test, [i + 10] * len(k))

        perm = np.random.permutation(1000*self.num_classes)
        x_temp = x_test.copy()
        y_temp = y_test.copy()
        for i in range(len(perm)):
            x_test[i] = x_temp[perm[i]]
            y_test[i] = y_temp[perm[i]]

        x_test = x_test.reshape(1000*self.num_classes, 28, 28, 1)

        """CONVERTING DATA"""
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print(x_train.shape, 'train samples')
        print(x_test.shape, 'test samples')

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        """self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28, 1)))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))"""

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        print(self.model.summary())
        from keras.utils import plot_model
        plot_model(self.model, to_file='model.png', show_shapes=True)
        history = self.model.fit(x_train, y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_data=(x_test, y_test))

        score = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        """from keras.utils import plot_model
        plot_model(self.model, to_file='model.png')"""

        print("SUMMARY")
        print(self.model.summary())

    def testImage(self, img):
        arr1 = np.asarray(img).reshape(1, 28, 28, 1) / 255
        res1 = self.model.predict(arr1)
        res1 = np.argmax(res1, axis=1)
        print(res1)