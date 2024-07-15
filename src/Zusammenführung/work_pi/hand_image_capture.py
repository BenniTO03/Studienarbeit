"""
Images are captured based on user input
Original images and cropped images are saved in folders
"""

import cv2
import os
import numpy as np
import math

from cvzone.HandTrackingModule import HandDetector


class HandImageCapture:
    """Camera-Klasse"""

    def __init__(self):
        """Instanziieren der Klasse mit dem Setzen der nötigen Variablen."""
        self.saved_image_path = None
        self.first_image_saved = False
        self.folder_name = None
        self.counter = 1
        self.crop_path = None

    def save_image(self, image, description):
        """Speichern des übergebenen Bildes mit der Beschreibung in einem Ordner.
        Die Bilder werden nach dem Schema 'COUNTER_BESCHREIBUNG.jpg' gespeichert. z.B.
        01_a.jpg oder 02_h.jpg
        :param image: Bild, welches gespeichert werden soll.
        :param description: String mit dem Buchstaben, welches das Bild darstellt."""
        # save image (not in a seperate function, because of the initialisation of picam)
        if not self.first_image_saved:
            self.folder_name = self.create_folder('./images')  # first folder
            self.first_image_saved = True  # set to true to start iterating the folders
        filename = f'{str(self.counter).zfill(2)}_{description}.jpg'  # filename with counter and label
        filepath = os.path.join(self.folder_name, filename)
        cv2.imwrite(filepath, image)

        self.saved_image_path = self.folder_name

        self.counter += 1

    def create_folder(self, folder_name):
        """Erstellen des Ordners, in dem die Bilder gespeichert werden. Diese Funktion ist notwendig,
        damit mehrere Durchläufe durchgeführt werden können. Somit werden die Bilder
        in Ordnern mit fortlaufenden Bildern gespeichert.
        Es wird überprüft, ob der Ordner bereits existiert, falls ja, wird
        der Counter um eines erhöht, bis der Ordner nicht existiert.
        :param folder_name: Ordnername, in dem die Dateien gespeichert werden sollen. Falls
        dieser Ordner bereits existiert, wird zusätzlich eine fortlaufende Zahl verwendet.
        :return: Ordnername als String"""
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

    def reset_values(self):
        """Setzt die Variablen zurück, damit ein weiterer Durchlauf durchgeführt werden kann."""
        self.saved_image_path = None
        self.first_image_saved = False
        self.folder_name = None
        self.counter = 1
        self.crop_path = None

    # Erkennen der Hände in den Bildern
    def get_hand(self):
        """Funktion zum Erkennen der Hände in den Bildern. Außerdem wird hier ein weiterer Ordner
        'hands_croped' erstellt, in dem die neuen Bilder gespeichert werden. Dort werden die Bilder
        mit gleichem Namen gespeichert.
        :return: Gibt den Pfad zurück, in dem die neuen Bilder gespeichert sind."""
        img_path = None
        if self.saved_image_path is not None:
            img_path = self.saved_image_path

        offset = 20
        img_size = 300
        files = os.listdir(img_path)

        for img in files:
            if img.endswith(".jpg"):
                imgpath = os.path.join(img_path, img)
                counter = None
                letter = None
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
                        x, y, w, h = hand['bbox']

                        # Bounding Box around hand
                        if y > offset and x > offset:
                            offset = 20
                        elif y < offset and x > offset:
                            offset = y - 1
                        elif y > offset and x < offset:
                            offset = x - 1

                        img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
                        img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                        aspect_ratio = h / w

                        # height > width -> vertical image
                        if aspect_ratio > 1:
                            k = img_size / h
                            w_cal = math.ceil(k * w)
                            img_resize = cv2.resize(img_crop, (w_cal, img_size))
                            img_resize_shape = img_resize.shape
                            w_gap = math.ceil((img_size - w_cal) / 2)

                            img_white[:, w_gap:w_cal + w_gap] = img_resize

                        # horizontal image
                        else:
                            k = img_size / w
                            h_cal = math.ceil(k * h)
                            img_resize = cv2.resize(img_crop, (img_size, h_cal))
                            img_resize_shape = img_resize.shape
                            h_gap = math.ceil((img_size - h_cal) / 2)

                            img_white[h_gap:h_cal + h_gap:] = img_resize
                        # save croped images in subfolder
                        crop_folder = "hands_croped"
                        self.crop_path = os.path.join(img_path, crop_folder)

                        if not os.path.exists(self.crop_path):
                            os.makedirs(self.crop_path)

                        new_file_name = f'{str(counter).zfill(2)}_{letter}.jpg'  # same letter as in original image
                        new_path = os.path.join(self.crop_path, new_file_name)

                        cv2.imwrite(new_path, img_white)

                    else:
                        print("No hands found.")

            else:
                print('file is no image')
                break

        return self.crop_path

