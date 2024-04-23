import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os, sys
from PIL import Image
from natsort import natsorted

from keras import layers


def flip(image_array):
    # spiegelt Hand
    random_flip = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical"),layers.RandomRotation(0.2),])
    random_flipped_array = random_flip(image_array)
    flipped_im = tf.keras.utils.array_to_img(random_flipped_array)
    return flipped_im

def zoom(image_array):
    # verwischt Teile des Bildes
    # heigt und width factor haben große Auswirkungen, die zur Unkenntlichkeit der Hand führen können
    random_zoom = tf.keras.Sequential([layers.RandomZoom(height_factor=0.2, width_factor=0.6)])
    random_zoom_array = random_zoom(image_array)
    zoomed_im = tf.keras.utils.array_to_img(random_zoom_array)
    return zoomed_im

def rotate(image_array):
    # dreht Hand um factor
    random_rotate = tf.keras.Sequential([layers.RandomRotation(factor=0.5)])
    random_rotate_array = random_rotate(image_array)
    rotated_im = tf.keras.utils.array_to_img(random_rotate_array)
    return rotated_im

def brightness(image_array):
    # verändert Helligkeit
    # hat keine großen Auswirkungen
    random_brightness = tf.keras.Sequential([layers.RandomBrightness(factor=0.1),])
    random_brightness_array = random_brightness(image_array)
    brightness_im = tf.keras.utils.array_to_img(random_brightness_array)
    return brightness_im

def channel_shift(image_array):
    # Verschiebung der Farbkanäle
    # sehr ähnlich zu brightness
    random_channel_shift = tf.keras.preprocessing.image.random_channel_shift(x=image_array, intensity_range=0.1)
    channel_shift_im = tf.keras.utils.array_to_img(random_channel_shift)
    return channel_shift_im

def shear(image_array):
    # zufällige Scherungstransformation
    random_shear = tf.keras.preprocessing.image.random_shear(x=image_array, intensity=1.6, row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode='nearest',
    cval=0.0,
    interpolation_order=1)
    shear_im = tf.keras.utils.array_to_img(random_shear)
    return shear_im

def shift(image_array):
    # Verschiebungstransformation
    random_shift = tf.keras.preprocessing.image.random_shift(x=image_array, wrg=0.2, hrg=0.2)
    shift_im = tf.keras.utils.array_to_img(random_shift)
    return shift_im



def get_images(images_path, prefix, start_number):
    count = 0
    for file in os.listdir(images_path):
        filepath = os.path.join(images_path, file)
        if filepath.lower().endswith(('.jpeg', '.jpg')):
            count += 1

            if count % 7 == 0:
                img = Image.open(filepath)
                image_array  = tf.keras.utils.img_to_array(img)

                # Art der Augmentation
                #save_image = shift(image_array)
                save_image = shear(image_array)
                #save_image = rotate(image_array)
                #save_image = flip(image_array)
                #save_image = zoom(image_array)

                destination_path = os.path.join(images_path, '{}_{:05d}.jpg'.format(prefix, start_number))
                save_image.save(destination_path)
                start_number += 1

if __name__ == "__main__":
    """
    Pfad ist Folder zu einzelnem Buchstaben
    Prefix muss für test und train Ordner angepasst werden
    start_number ist nächsthöhere Zahl des letzten files
    in get_images muss Art der Augmentation angegeben werden
    """
    images_path = '../02_data_crop/train/26'
    prefix = 'z'
    start_number = 162
    get_images(images_path, prefix, start_number)
