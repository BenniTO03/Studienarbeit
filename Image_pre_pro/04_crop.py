from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import cv2
import os

"""
     Detektion der Hand im Bild
     entsprechendes Zuschneiden des Bildes
     alte Bilder werden gelöscht und zugeschnittene Bilder im gleichen Ordner gespeichert

     anpassen:
     - source_path
"""


def Crop_Images_Hands(input_folder):

     files = os.listdir(input_folder)
     zaehler = 1
     imgSize = 300
     offset = 20

     try:
         for file in files:
               img_path = os.path.join(input_folder, file)
               image = cv2.imread(img_path)

               if image is None:
                    print("Image is None")
                    continue
               #img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

               detector = HandDetector(maxHands=1)
               hands, img = detector.findHands(image)

               if hands:
                    # Detektion nur einer Hand
                    hand = hands[0]
                    x,y,w,h = hand['bbox']
                    
                    # Bounding Box um Hand
                    if y > offset and x > offset:
                         offset = 20
                         
                    elif y < offset and x > offset:
                         offset = y - 1

                    elif y > offset and x < offset:
                         offset = x - 1
                    
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
               
                    os.remove(img_path)
                    new_file_name = f'{str(zaehler).zfill(5)}.jpg'
                    new_path = os.path.join(input_folder, new_file_name)

                    cv2.imwrite(new_path, imgWhite)
                    zaehler += 1

               else:
                    print("No hands found.")

     except Exception as e:
          print(str(e))

source_path = '../02_data_crop/test/26'
Crop_Images_Hands(input_folder=source_path)