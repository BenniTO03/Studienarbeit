import os
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import models
import keras
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.utils import to_categorical 

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class Data:

    def __init__(self, images):
        self.images_path = images


    def get_images(self):

        array_images = []
        train_or_test_folder = os.listdir(self.images_path)

        for folder in natsorted(train_or_test_folder):
            single_folder = os.path.join(self.images_path, folder)

            for file in os.listdir(single_folder):
                filepath = os.path.join(single_folder, file)

                if filepath.lower().endswith(('.jpeg', '.jpg')):
                    image = tf.io.read_file(filepath)
                    tensor_image = tf.io.decode_image(image, channels=1, dtype=tf.dtypes.float32)
                    tensor_image = tf.image.resize(tensor_image, [250, 250])

                    # Array von TensorFlow Tensoren
                    array_images.append(tensor_image)
        
        return array_images
    
    
    def get_label(self):

        array_label = []
        for folder in natsorted(os.listdir(self.images_path)):
            label = int(folder)

            for file in os.listdir(os.path.join(self.images_path, folder)):
                array_label.append(label)
            
        return array_label




class Data_preparation:

    def __init__(self):
        self.data_train = Data('cnn/resized_images/train')
        self.data_test = Data('cnn/resized_images/test')
 

    def get_data_as_array(self):
        train_array_images = self.data_train.get_images()
        train_array_labels = self.data_train.get_label()

        test_array_images = self.data_test.get_images()
        test_array_labels = self.data_test.get_label()

        return train_array_images, train_array_labels, test_array_images, test_array_labels


    def get_data_as_numpy_array(self, train_array_images, train_array_labels, test_array_images, test_array_labels):
        train_numpy_images = np.array([tensor.numpy() for tensor in train_array_images])
        train_numpy_labels = np.array([tensor.numpy() for tensor in train_array_labels])

        test_numpy_images = np.array([tensor.numpy() for tensor in test_array_images])
        test_numpy_labels = np.array([tensor.numpy() for tensor in test_array_labels])

        return train_numpy_images, train_numpy_labels, test_numpy_images, test_numpy_labels
    

    def prep_images(self, train_numpy_images, test_numpy_images):
        # Laden und Vorverarbeiten der Daten
        train_normalized_images = train_numpy_images.astype('float32')
        train_normalized_images /= 255

        test_normalized_images = test_numpy_images.astype('float32')
        test_normalized_images /= 255

        NrTrainimages = train_normalized_images.shape[0]
        NrTestimages = test_normalized_images.shape[0]

        return train_normalized_images,  test_normalized_images


    def prep_label(self, train_numpy_labels, test_numpy_labels):
        train_binary_labels = to_categorical(train_numpy_labels)
        test_binary_labels = to_categorical(test_numpy_labels)

        return train_binary_labels, test_binary_labels




class Model:

    def __init__(self, train_images, train_labels, test_images, test_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

        self.model = self.initialize()
        self.compiled_model = self.compile()


    def initialize(self):
        # Netzwerkarchitektur
        num_classes = 27
        model = Sequential()

            # Detektion
        model.add(Conv2D(32, kernel_size=(5,5), activation= 'relu', input_shape=(250, 250))) # input shape der images

            # Conv_Block 2
        model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))

            # Identifikation
        model.add(Flatten())
        model.add(Dense(128, activation='relu', name='features'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax')) # Anzahl der Klassen als Ausgabedimension angeben
        model.summary()

        return model
    

    def compile(self):
        # Festlegung der Verlustfunktion und Optimierung
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return self.model
    

    def train(self):

        datagen = ImageDataGenerator()
        train_generator = datagen.flow(self.train_images, self.train_labels, batch_size=32)

        batch_size = 32
        epochs = 10

        history = self.compiled_model.fit(train_generator,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(self.test_images, self.test_labels))
        
        return history
        

    def evaluate(self):
        # Evaluation
        score = self.compiled_model.evaluate(self.test_images, self.test_labels)
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])


    def save(self):
        self.compiled_model.save('cnn/Models/Model_2.h5')



if __name__ == "__main__":
    data_prep = Data_preparation()
    train_array_images, train_array_labels, test_array_images, test_array_labels = data_prep.get_data_as_array()
    train_numpy_images, train_numpy_labels, test_numpy_images, test_numpy_labels = data_prep.get_data_as_numpy_array(train_array_images, train_array_labels, test_array_images, test_array_labels)
    
    train_normalized_images, test_normalized_images = data_prep.prep_images(train_numpy_images, test_numpy_images)
    train_binary_labels, test_binary_labels = data_prep.prep_label(train_numpy_labels, test_numpy_labels)

    cnn = Model(train_normalized_images, train_binary_labels, test_normalized_images, test_binary_labels)
    cnn.compile()
    cnn.train()
    cnn.evaluate()
    cnn.save()

