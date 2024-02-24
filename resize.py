from PIL import Image
import os

images_path = "./train"
target_size = (256, 256)
# get images as tensors
for folder in os.listdir(images_path):
    label = folder   # a,b,c,d, ...

    for file in os.listdir(os.path.join(images_path, folder)):
        path = os.path.join(images_path, folder, file)
        img = Image.open(path)
        img_resized = img.resize(target_size)
        img_resized.save(path)