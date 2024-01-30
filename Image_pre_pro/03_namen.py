import os
import shutil

"""
Kopiert neu generierte Bilder in den kompletten Datensatz

anpassen:
  - target_folder
  - images_folder
  - prefix  -> für den jeweiligen Buchstaben/Ordner
  - start_number  -> damit es keine Duplikate gibt bei nächstgrößer Zahl anfangen, die schon vergeben ist
"""

target_folder = '../02_data_crop/train/z'
images_folder = '../Bilder_Benni/Z'

prefix = 'z'
start_number = 67


for file in os.listdir(images_folder):
    if file.endswith('.jpg'):

        source_path = os.path.join(images_folder, file)
        destination_path = os.path.join(target_folder, '{}_{:05d}.jpg'.format(prefix, start_number))

        shutil.copy(source_path,destination_path)

        start_number += 1