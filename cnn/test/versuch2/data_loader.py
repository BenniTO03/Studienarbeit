import os
import cv2
from natsort import natsorted
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import keras
from sklearn.model_selection import train_test_split

class Data():

    def __init__(self):
        pass

    @staticmethod

    def get_images(images_path):
    # speichert Bilder als numpy array

        array_images = []
        train_or_test_folder = os.listdir(images_path)

        for folder in natsorted(train_or_test_folder):
            single_folder = os.path.join(images_path, folder)

            for file in os.listdir(single_folder):
                filepath = os.path.join(single_folder, file)

                if filepath.lower().endswith(('.jpeg', '.jpg')):
                    image = cv2.resize(cv2.imread(filepath), (64, 64))  # resize Größe bestimmt durch vortainiertes Netz
                    array_images.append(image)

        images = np.array(array_images)
        
        return images
    
    @staticmethod    
    def get_label(images_path):
    # speichert Lables als numpy array

        array_label = []
        for folder in natsorted(os.listdir(images_path)):
            label = int(folder)

            for file in os.listdir(os.path.join(images_path, folder)):
                array_label.append(label)
        labels = np.array(array_label)
            
        return labels
    
    @staticmethod
    def preprocess(images, labels, X_eval, y_eval):
        images, X_eval = images / 255.0, X_eval / 255.0
        labels = keras.utils.to_categorical(labels, num_classes=27)
        y_eval = keras.utils.to_categorical(y_eval, num_classes=27)
        return images, labels, X_eval, y_eval
    
    @staticmethod
    def split(images, labels):
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, stratify = labels)
        return X_train, X_test, y_train, y_test




class PyTorchDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image).permute(2, 0, 1)

        return image, label


class Loader():
    def __init__(self):
        pass

    @staticmethod

    def load_data():

        images = Data.get_images("../../02_data_crop/train")
        labels = Data.get_label("../../02_data_crop/train") # train images

        X_eval = Data.get_images("../../02_data_crop/test")
        y_eval = Data.get_label("../../02_data_crop/test")  # Evaluierungs Bilder

        images, labels, X_eval, y_eval = Data.preprocess(images, labels, X_eval, y_eval)
        X_train, X_test, y_train, y_test = Data.split(images, labels)
        print(X_train.shape)



        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        training_dataset = PyTorchDataset(images=X_train, labels=y_train, transform=transforms.Compose([transforms.ToTensor()])) #transforms.Normalize(mean=mean, std=std)]))
        test_dataset = PyTorchDataset(images=X_test, labels=y_test, transform=transforms.Compose([transforms.ToTensor()]))  #transforms.Normalize(mean=mean, std=std)]))
        eval_dataset = PyTorchDataset(images=X_eval, labels=y_eval, transform=transforms.Compose([transforms.ToTensor()]))  #transforms.Normalize(mean=mean, std=std)]))
        image_train, label_train = training_dataset[1]
        shape = image_train.shape
        print('training_dataset image', shape)

        return training_dataset, test_dataset, eval_dataset

                                                                                    