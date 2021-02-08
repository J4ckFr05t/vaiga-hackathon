from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
import numpy as np
from PIL import Image
from tensorflow import keras
opt = keras.optimizers.Adam(learning_rate=0.01)
Image.MAX_IMAGE_PIXELS = 1000000000 #To avoid Decompression Bomb Warning


# 25 classes in the Dataset
path_root = "pest/train"
num_classes = 12


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
        self.Malware_model.add(Dense(1024, activation='relu'))
        self.Malware_model.add(Dropout(0.4))
        self.Malware_model.add(Dense(512, activation='relu'))
        self.Malware_model.add(Dense(num_classes, activation='softmax'))
        self.Malware_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def train_model(self):
        # Train and test our model
        self.Malware_model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=20,
                               batch_size=64, verbose=1)
        self.Malware_model.save("vaigamodel-pest")
        scores = self.Malware_model.evaluate(self.x_test, self.y_test)
        self.Malware_model.summary()
        print("\nAccuracy: ", scores[1])
        print("\nLoss : ", scores[0])
        

    def predictor(self):
        model = keras.models.load_model('vaigamodel-plant')
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
