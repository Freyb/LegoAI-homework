import keras
from keras.datasets import mnist
import numpy as np
from src.Pipeline import Pipeline
import matplotlib.pyplot as plt


class Model():
    """
    The Model class is responsible for building the network up
    and merging augmented data with MNIST dataset
    """
    def __init__(self):
        self.model = None
        self.augmented_test_images = None
        self.augmented_train_images = None
        self.num_classes = 14
        self.batch_size = 256
        self.epochs = 30

    def gen_data(self):
        """
        Builds up Pipeline with Elastic Distorsions.
        Samples data after transformations.
        :return: None
        """
        print("Start generating augmented images")
        p = Pipeline("./synthetic")
        p.elastic_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=1)
        p.elastic_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=1)
        p.elastic_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=1)
        self.augmented_train_images = p.sample(6000)
        self.augmented_test_images = p.sample(1000)

    def train(self):
        """
        Merges augmented data with MNIST dataset, then shuffles and converts the full dataset.
        Builds up the model then trains it.
        :return: None
        """

        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)

        # Merge and shuffle train data
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

        # Merge and shuffle test data
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

        # Convert data
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # Build Model
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.6))
        self.model.add(Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.6))
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

        # Print Model summary and render model.png
        """print(self.model.summary())
        from keras.utils import plot_model
        plot_model(self.model, to_file='model.png', show_shapes=True)"""

        # Train Model
        history = self.model.fit(x_train, y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_split=0.3,
                                 shuffle=True)

        # summarize history for loss
        plot_title = "64x5c2m0.6d - 128x5c2m0.6d - 1024d0.4d"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(plot_title)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train: '+str(history.history['loss'][-1]), 'valid: '+str(history.history['val_loss'][-1])], loc='upper left')
        plt.show()

        # Evaluate Model
        score = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        """import time
        import winsound
        time.sleep(1)
        for i in range(3):
            time.sleep(1)
            winsound.Beep(500, 1000)"""

    def testImage(self, img):
        output_vector = [i for i in range(10)] + ['X', '-', '+', '/']
        img_array = np.asarray(img).reshape(1, 28, 28, 1) / 255
        res_vector = self.model.predict(img_array)
        res = np.argmax(res_vector, axis=1)
        print("Preditction:", output_vector[res[0]])
        for i in range(14):
            print(output_vector[i], ":", res_vector[0][i])
