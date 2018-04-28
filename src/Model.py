import keras
from keras.datasets import mnist
import numpy as np
from PIL import Image
from src.Pipeline import Pipeline


class Model():
    def __init__(self):
        self.model = None
        self.augmented_test_images = None
        self.augmented_train_images = None
        self.num_classes = 11
        self.batch_size = 256
        self.epochs = 10

    def gen_data(self):
        print("Start generating augmented images")
        p = Pipeline("./synthetic")
        p.random_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=1)
        p.random_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=1)
        p.random_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=1)
        self.augmented_train_images = p.sample(6000)
        self.augmented_test_images = p.sample(1000)

        """images = []
        im = Image.open("synt_plus.png")
        im = im.convert('L')
        images.append(im)

        print("Start generating augmented images")
        self.augmented_train_images = d.perform_operation(images, 6000)
        self.augmented_test_images = d.perform_operation(images, 1000)
        for i in range(20):
            print("save:")
            Image.fromarray(self.augmented_train_images[0, i].reshape(28, 28)).convert('L').save('test'+str(i)+'.png');"""

    def train(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        print()
        #Image.fromarray(self.augmented_train_images[0][0].reshape(28, 28)).convert('L').save('look4.png')

        for i, k in enumerate(self.augmented_train_images):
            x_train = np.concatenate((x_train, k), axis=0)
            y_train = np.append(y_train, [i + 10] * len(k))

        #Image.fromarray(x_train[60000].reshape(28, 28)).convert('L').save('look6.png')

        perm = np.random.permutation(66000)
        x_temp = x_train.copy()
        y_temp = y_train.copy()
        vvv = False
        for i in range(len(perm)):
            x_train[i] = x_temp[perm[i]]
            y_train[i] = y_temp[perm[i]]
            if y_temp[i] == 10 and not vvv:
                print("i:", i, "perm:", perm[i])
                #Image.fromarray(x_temp[i].reshape(28, 28)).convert('L').save('look.png')
                vvv = True

        for i, k in enumerate(self.augmented_test_images):
            x_test = np.concatenate((x_test, k), axis=0)
            y_test = np.append(y_test, [i + 10] * len(k))

        perm = np.random.permutation(11000)
        x_temp = x_train.copy()
        y_temp = y_train.copy()
        for i in perm:
            x_test[i] = x_temp[perm[i]]
            y_test[i] = y_temp[perm[i]]

        print(x_train)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        print(y_train.shape)

        from keras.models import Sequential
        from keras.layers import Dense, Dropout

        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=(784,)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

        history = self.model.fit(x_train, y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_data=(x_test, y_test))

        score = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def testImage(self, img):
        arr1 = np.asarray(img).reshape(1, 784) / 255
        res1 = self.model.predict(arr1)
        res1 = np.argmax(res1, axis=1)
        print(res1)
