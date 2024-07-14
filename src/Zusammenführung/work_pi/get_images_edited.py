"""
Images are captured based on user input
Original images and cropped images are saved in folders
"""

import cv2
import os
import numpy as np
import math
import time

from cvzone.HandTrackingModule import HandDetector


class Camera():
    def __init__(self):
        self.first_image_saved = False
        self.folder_name = None
        self.counter = 1
        self.crop_path = None
        
    # Erstellen des Ordners und speichern des übergebenen Bildes mit Beschreibung
    def save_image(self, image, description):
        # save image (not in a seperate function, because of the initialisation of picam)
        if not self.first_image_saved:
            self.folder_name = self.create_folder('./images')  # first folder
            self.first_image_saved = True  # set to true to start iterating the folders
        filename = f'{str(self.counter).zfill(2)}_{description}.jpg' # filename with counter and label
        filepath = os.path.join(self.folder_name, filename)
        #picam.capture_file(filepath) # save image
        cv2.imwrite(filepath, image)

        self.saved_image_path = self.folder_name

        self.counter += 1

    # Erstellen der Bilder-Ordners mit fortlaufenden Zahlen (z.B. bei mehreren Durchläufen)
    def create_folder(self, folder_name):
        # checks wether folder already exists
        if not os.path.exists(folder_name):  
            os.makedirs(folder_name)
            return folder_name
        else: # creates folders with ascending numbering
            i = 1
            while True:
                new_folder_name = f'{folder_name}_{str(i).zfill(2)}'
                if not os.path.exists(new_folder_name):
                    os.makedirs(new_folder_name)
                    return new_folder_name
                i += 1
                
    def reset_values(self):
        self.first_image_saved = False
        self.folder_name = None
        self.counter = 1
        self.crop_path = None
    
    # Erkennen der Hände in den Bildern
    def get_hand(self):
        # find hand in image and crops it accordingly
        if self.saved_image_path is not None:
            img_path = self.saved_image_path

        offset = 20
        imgSize = 300
        files = os.listdir(img_path)

        for img in files:
            if img.endswith(".jpg"):
                imgpath = os.path.join(img_path, img)
                
                try:
                    counter = img.split('_')[0]
                    letter = img.split('_')[1].split('.')[0]
                except IndexError:
                    print(f"Fehler beim Extrahieren des Buchstabens aus dem Dateinamen: {img}")

                img = cv2.imread(imgpath)
                if img is None:
                    print("Fehler beim Laden des Bildes: {imgpath}")
                else:
                    detector = HandDetector(maxHands=1)
                    hands, img = detector.findHands(img)

                    if hands:
                        hand = hands[0]
                        x,y,w,h = hand['bbox']
                        
                        # Bounding Box around hand
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
                
                        # height > width -> vertical image
                        if aspectRatio > 1:
                                k = imgSize/h
                                wCal = math.ceil(k * w)
                                imgResize = cv2.resize(imgCrop,(wCal, imgSize))
                                imgResizeShape = imgResize.shape
                                wGap = math.ceil((imgSize-wCal) / 2)

                                imgWhite[:, wGap:wCal+wGap] = imgResize
                    
                        # horizontal image
                        else:
                                k = imgSize/w
                                hCal = math.ceil(k * h)
                                imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                                imgResizeShape = imgResize.shape
                                hGap = math.ceil((imgSize-hCal) / 2)

                                imgWhite[hGap:hCal+hGap:] = imgResize
                        # save croped images in subfolder
                        crop_folder = "hands_croped"
                        self.crop_path = os.path.join(img_path, crop_folder)
                        
                        if not os.path.exists(self.crop_path):
                            os.makedirs(self.crop_path)

                        new_file_name = f'{str(counter).zfill(2)}_{letter}.jpg' # same letter as in original image
                        new_path = os.path.join(self.crop_path, new_file_name)

                        cv2.imwrite(new_path, imgWhite)
                        #counter += 1

                    else:
                        print("No hands found.")
                        
            else:
                print('file is no image')
                break
                
        return self.crop_path # gib den Pfad zu den neuen Bildern zurück
