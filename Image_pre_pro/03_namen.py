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

target_folder = './data_test/A'
images_folder = './data_new/0'

prefix = 'A'
start_number = 10


for file in os.listdir(images_folder):
    if file.endswith('.jpg'):

        source_path = os.path.join(images_folder, file)
        destination_path = os.path.join(target_folder, '{}_{:05d}.jpg'.format(prefix, start_number))

        shutil.copy(source_path,destination_path)

        start_number += 1