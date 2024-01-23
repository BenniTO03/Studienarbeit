from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import math
import cv2
import os
import time

folder = './data_both/A'

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)


counter = 0
imgSize = 300
offset = 20


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
   
    if hands:
            # Detektion nur einer Hand
            hand = hands[0]
            x,y,w,h = hand['bbox']
                        
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
            
            cv2.imshow("Image", imgCrop)
            cv2.imshow("Image White", imgWhite)
                

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord('s'):
            print("I am saving.")
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(counter)