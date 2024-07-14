import cv2
import os
import numpy as np
import math

from cvzone.HandTrackingModule import HandDetector

from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F


class Kamera():
    def __init__(self):
        self.first_image_saved = False
        self.folder_name = None

    """
    def camera_module_pi(self):
        picam = Picamera2()
        config = picam.create_preview_configuration()
        config.format = 'RGB'
        picam.configure(config)
        picam.start_preview(Preview.QTGL)
        picam.start()

        while True:
            description = input("Gib eine Beschreibung für das Bild ein: ")
            time.sleep(2)
            picam.capture_file(f'{description}.jpg')

            key = input("Möchtest du ein weiteres Bild aufnehmen? Drücke 'q' zum Beenden oder eine beliebige Taste für ein neues Bild: ")

            if key == 'q':
                break

            picam.stop()
    """
     
    def camera_module_csv(self):
        print("Die Kamera wird nun gestartet.")
        cap = cv2.VideoCapture(0)
        counter = 1

        while True:
            ret, image = cap.read()
            cv2.putText(image, 'Ready? Press "s" ! :)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Kamera', image)

            if cv2.waitKey(25) == ord('s'):  # muss auf Bild geklickt werden
                description = input("Gib den Buchstaben für das Bild ein, das du zeigen möchtest: ") # muss in Terminal eingegeben werden
                self.save_images(image, description, counter)
                counter += 1

                while True:
                    choice = input("Möchtest du ein weiteres Bild aufnehmen? Drücke 's' für ja und 'q' für nein: ")
                    if choice == 's':
                        break
                    elif choice == 'q':
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    else:
                        print("Ungültige Eingabe. Bitte versuche es erneut.")

            elif cv2.waitKey(25) == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
         

    def take_image(self):
        # Zugriff auf Rasperry Pi Kamera
        print("Drücke die Taste 's', um ein Bild aufzunehmen.")
        print("Drücke die Taste 'q', um das Programm zu beenden.")

        
        self.camera_module_csv()
        self.get_hand()

    
    def create_folder(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

            return folder_name
        else:
            i = 1
            while True:
                new_folder_name = f'{folder_name}_{str(i).zfill(2)}'
                if not os.path.exists(new_folder_name):
                    os.makedirs(new_folder_name)
                    return new_folder_name
                i += 1


    def save_images(self, image, label , counter):
        if not self.first_image_saved:
            self.folder_name = self.create_folder('./images')
            self.first_image_saved = True
        filename = f'{str(counter).zfill(2)}_{label}.jpg'
        filepath = os.path.join(self.folder_name, filename)
        cv2.imwrite(filepath, image)

        self.saved_image_path = self.folder_name

    
    def get_hand(self):

        if self.saved_image_path is not None:
            img_path = self.saved_image_path

        offset = 20
        imgSize = 300
        counter = 1
        print(img_path)
        files = os.listdir(img_path)

        for img in files:
            if img.endswith(".jpg"):
                imgpath = os.path.join(img_path, img)

                img = cv2.imread(imgpath)
                if img is None:
                    print("Fehler beim Laden des Bildes")
                else:

                    letter = img.split('_')[1].split('.')[0]
                    detector = HandDetector(maxHands=1)
                    hands, img = detector.findHands(img)

                    if hands:
                        hand = hands[0]
                        x,y,w,h = hand['bbox']
                        
                        # Bounding Box um Hand
                        if y > offset and x > offset:
                                offset = 20
                        elif y < offset and x > offset:
                                offset = y - 1
                        elif y > offset and x < offset:
                                offset = x - 1
                        
                        imgCrop = img[y - offset:y + h + offset, x - offset:x + w +offset]

                        imgWhite = np.ones((imgSize,imgSize,3),np.uint8) * 255
                        imgCrop = img[y - offset:y + h + offset, x - offset:x + w +offset]

                        aspectRatio = h/w
                
                        # Höhe > Breite -> vertikales Bild
                        if aspectRatio > 1:
                                k = imgSize/h
                                wCal = math.ceil(k * w)
                                imgResize = cv2.resize(imgCrop,(wCal, imgSize))
                                imgResizeShape = imgResize.shape
                                wGap = math.ceil((imgSize-wCal) / 2)

                                imgWhite[:, wGap:wCal+wGap] = imgResize
                    
                        # horizontales Bild
                        else:
                                k = imgSize/w
                                hCal = math.ceil(k * h)
                                imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                                imgResizeShape = imgResize.shape
                                hGap = math.ceil((imgSize-hCal) / 2)

                                imgWhite[hGap:hCal+hGap:] = imgResize

                        crop_folder = "hands_croped"
                        crop_path = os.path.join(img_path, crop_folder)
                        
                        if not os.path.exists(crop_path):
                            os.makedirs(crop_path)

                        new_file_name = f'{str(counter).zfill(2)}_{letter}.jpg'
                        new_path = os.path.join(crop_path, new_file_name)

                        cv2.imwrite(new_path, imgWhite)
                        counter += 1

                    else:
                        print("No hands found.")

            else:
                print('file is no image')
                break

            
class ImageClassificationBase(nn.Module):
    
        def training_step(self, batch):
            """ 
                calculates trainigs loss and accuracy
            """ 
            images, labels = batch # extracts images and labels from batch
            out = self(images)     # model is applied to the images
            train_loss = F.cross_entropy(out, labels) # calculates cross_entropy loss between predictions and labels
            train_acc = accuracy(out, labels) # calculates accuracy between predictions and labels
            return train_loss, train_acc
        
        def validation_step(self, batch):
            """ 
                calculates validation loss and accuray
                equivalent to training_step
            """ 
            images, labels = batch 
            out = self(images)                    
            loss = F.cross_entropy(out, labels)   
            acc = accuracy(out, labels)
            return {'val_loss': loss.detach(), 'val_acc': acc}  # loss will not calculated further as it is only for evualuation
            
        def validation_epoch_end(self, outputs):
            """ 
                combines loss and accuracy per batch for one epoch
            """
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            """ 
                defines output per epoch
            """
            print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))


class NaturalSceneClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1: Convolutional Layer with 3 input channels and 32 output channels
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),

            # Layer 2: Convolutional Layer with 32 input channels and 64 output channels
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            # Layer 3: Max Pooling Layer with kernel size 2x2 and Stride 2x2
            nn.MaxPool2d(2,2),
        
            # Layer 4: Convolutional Layer with 64 input channels and 128 output channels
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            # Layer 5: Convolutional Layer with 128 input channels and 128 output channels
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            # Layer 6: Max Pooling Layer with kernel size 2x2 and Stride 2x2
            nn.MaxPool2d(2,2),
            
            # Layer 7: Convolutional Layer with 128 input channels and 256 output channels
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            # Layer 8: Convolutional Layer with 256 input channels and 256 output channels
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            # Layer 9: Max Pooling Layer with kernel size (2x2) and Stride (2x2)
            nn.MaxPool2d(2,2),
            
            # Layer 10: Flatten Layer to transform data into vector
            nn.Flatten(),

            # Layer 11: Fully Connected Layer with input size (16384) and output size (1024)
            nn.Linear(16384,1024),  # last value = first value of the next layer
            nn.ReLU(),

            # Layer 12: Fully Connected Layer with input size (1024) and output size (512)
            nn.Linear(1024, 512),   # last value = first value of the next layer
            nn.ReLU(),

            # Layer 13: Fully Connected Layer with input size (512) and output size (26)
            nn.Linear(512,26)  # number of classes
        )
    
    def forward(self, xb):
        return self.network(xb)


class Model():
    label_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 
                   19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25: 'z'}
    def __init__(self, model_path, data_dir):

        self.model_path = model_path
        self.data_dir = data_dir

    def prediction(self, model, data_dir):

        transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
        # run through all images in the folder
        for image_name in os.listdir(data_dir):
            image_path = os.path.join(data_dir, image_name)
            image = Image.open(image_path)
            image_tensor = transform(image).unsqueeze(0)

            output = model(image_tensor)  # put images into model

            predicted_class = torch.argmax(output).item()  # get predicted label
            letter_label = Model.label_to_letter[predicted_class]  # change predicted number to predicted letter

            print(f'Das Bild {image_name} wurde der Klasse {letter_label} zugeordnet.')

    def get_word(self):
        # put single imagaes into word
        pass

    def check_prediction(self):
        # stimme es mit USer input ab
        pass
    
    def correct_word(self):
        # korrigiere, falls falsch
        pass

    def predict_image(self):

        model = torch.load(self.model_path)


        model.eval()
        self.prediction(model, self.data_dir)







if __name__ == '__main__':
    camera = Kamera()
    camera.take_image()

    model_path = './cnn/cnn2/versuch2/output/model_epoch15.pth'
    data_dir = './images/'
    model = Model(model_path, data_dir)
    model.predict_image()
