# Bilder werden aufgenommen nach Beuntzereingabe
# originale Bilder und zugeschnittene Bilder werden in Ordnern gespeichert

import cv2
import os
import numpy as np
import math

from cvzone.HandTrackingModule import HandDetector
import time
from picamera2 import Picamera2, Preview
from libcamera import controls



class Kamera():
    def __init__(self):
        self.first_image_saved = False
        self.folder_name = None

    def camera_module_pi(self):
        print("Die Kamera wird nun gestartet.")
        picam = Picamera2()  # object to reference the module and control the camera
        counter = 1

        while True:
            config = picam.create_preview_configuration()
            picam.configure(config)
            picam.start_preview(Preview.QTGL)  # start the preview window

            print("Ready? Press "s" ! :)")
            picam.start()

            if cv2.waitKey(25) == ord('s'):
                description = input("Gib den Buchstaben für das gezeigte Bild ein: ")
                # save image
                
                if not self.first_image_saved:
                    self.folder_name = self.create_folder('./images')
                    self.first_image_saved = True
                filename = f'{str(counter).zfill(2)}_{description}.jpg'
                filepath = os.path.join(self.folder_name, filename)
                picam.capture(filepath)

                self.saved_image_path = self.folder_name

                counter += 1

                while True:
                     choice = input("Möchtest du ein weiteres Bild aufnehmen? Drücke 's' für ja und 'q' für nein: ")
                     if choice == 's':
                          break
                     elif choice == 'q':
                          picam.stop_preview()
                          picam.stop()
                          return
                     else:
                          print("Ungültige Eingabe.")
            elif cv2.waitKey(25) == ord('q'):
                picam.stop_preview()
                picam.stop()
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

                try:
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

            


if __name__ == '__main__':
    camera = Kamera()
    camera.take_image()
