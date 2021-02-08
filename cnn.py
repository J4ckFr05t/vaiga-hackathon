from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
import numpy as np
from PIL import Image
from tensorflow import keras
opt = keras.optimizers.Adam(learning_rate=0.01)
Image.MAX_IMAGE_PIXELS = 1000000000 #To avoid Decompression Bomb Warning


# 25 classes in the Dataset
path_root = "../../Downloads/all/train"
num_classes = 24


class CNN:
    def __init__(self):
        # Generating DataSet
        self.Malware_model = Sequential()
        batches = ImageDataGenerator().flow_from_directory(directory=path_root, target_size=(64, 64), batch_size=22000)
        self.imgs, self.labels = next(batches)

        # Split into train and test
        print("\nSplitting Dataset\n")
        x_train, x_test, y_train, y_test = train_test_split(self.imgs / 255, self.labels, test_size=0.3)

        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)

    def cnn_model(self):
        self.Malware_model.add(Conv2D(30, kernel_size=(3, 3),
                                      activation='relu',
                                      input_shape=(64, 64, 3)))
        self.Malware_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.Malware_model.add(Conv2D(15, (3, 3), activation='relu'))
        self.Malware_model.add(Conv2D(15, (3, 3), activation='relu'))
        self.Malware_model.add(BatchNormalization())
        self.Malware_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.Malware_model.add(Conv2D(8, (3, 3), activation='relu'))
        self.Malware_model.add(Conv2D(8, (3, 3), activation='relu'))
        self.Malware_model.add(BatchNormalization())
        self.Malware_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.Malware_model.add(Dropout(0.25))
        self.Malware_model.add(Flatten())
        self.Malware_model.add(Dense(512, activation='relu'))
        self.Malware_model.add(Dropout(0.4))
        self.Malware_model.add(Dense(128, activation='relu'))
        self.Malware_model.add(Dense(num_classes, activation='softmax'))
        self.Malware_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def train_model(self):
        # Train and test our model
        self.Malware_model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=20,
                               batch_size=64, verbose=1)
        self.Malware_model.save("vaigamodel-plant")
        scores = self.Malware_model.evaluate(self.x_test, self.y_test)
        self.Malware_model.summary()
        print("\nAccuracy: ", scores[1])
        print("\nLoss : ", scores[0])

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        # Train and validation accuracy
        plt.plot(epochs, acc, 'b', label='Training accurarcy')
        plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
        plt.title('Training and Validation accurarcy')
        plt.legend()

        plt.figure()

        # Train and validation loss
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.legend()
        plt.show()
        

    def predictor(self):
        model = keras.models.load_model('vaigamodel-pest')
        test = ImageDataGenerator().flow_from_directory(
            directory="pred", target_size=(64, 64), batch_size=1)
        timg, label = next(test)
        result = model.predict(timg)
        print(result)



if __name__ == '__main__':
    cnn1 = CNN()
    cnn1.cnn_model()
    cnn1.train_model()
    #cnn1.predictor()
