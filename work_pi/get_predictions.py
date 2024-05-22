import os

from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F


class ImageClassificationBase(nn.Module):
    
        def training_step(self, batch):
            images, labels = batch
            out = self(images)
            train_loss = F.cross_entropy(out, labels)
            train_acc = accuracy(out, labels)
            return train_loss, train_acc
        
        def validation_step(self, batch):
            images, labels = batch 
            out = self(images)                    
            loss = F.cross_entropy(out, labels)   
            acc = accuracy(out, labels)
            return {'val_loss': loss.detach(), 'val_acc': acc}
            
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))


class NaturalSceneClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(16384,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,26)
        )
    
    def forward(self, xb):
        return self.network(xb)


class Model():
    # translation dictionary between label and letter
    label_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 
                   19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25: 'z'}
    
    def __init__(self, model_path, data_dir):  # must be spedified when calling the class
        self.model_path = model_path
        self.data_dir = data_dir


    def prediction(self, model, data_dir):
        # iterates over images in folder and makes predictions using the model
        transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])  # same transformations as in the model
        predicted_letters = []
        for image_name in os.listdir(data_dir):
            image_path = os.path.join(data_dir, image_name)
            image = Image.open(image_path)
            image_tensor = transform(image).unsqueeze(0) # add channel

            output = model(image_tensor)  # put images into model

            predicted_class = torch.argmax(output).item()  # get predicted label
            predicted_letter = Model.label_to_letter[predicted_class]  # change predicted number to predicted letter

            print(f'Das Bild {image_name} wurde der Klasse {predicted_letter} zugeordnet.')
            predicted_letters.append(predicted_letter)

        return predicted_letters


    def predict_image(self):
        model = torch.load(self.model_path)
        model.eval() # set model in evaluation mode
        predicted_letters = self.prediction(model, self.data_dir) # get all predicted letters
        return predicted_letters