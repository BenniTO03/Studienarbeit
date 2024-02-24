import tensorflow as tf
import os
import json

def get_images(images_path):

    images_dirs = os.listdir(images_path)   # folder of single buchstabe   -> a,b,c,d,e,f,g,h,i,j,k, ...
    tensor_img = []   
    tensor_l = []

    # get images as tensors
    for folder in os.listdir(images_path):
        label = folder   # a,b,c,d, ...

        for file in os.listdir(os.path.join(images_path, folder)):
            path = os.path.join(images_path, folder, file)
 
            if file.lower().endswith(('.jpeg', '.jpg')):
                image = tf.io.read_file(path)
                tensor_image = tf.io.decode_image(image, channels=3, dtype=tf.dtypes.float32)
                tensor_image = tf.image.resize(tensor_image, [256, 256])
                tensor_img.append(tensor_image)
                tensor_l.append(label)
                tensor_l = list(map(int,tensor_l))
            else: 
                print("No images in folder.")
    
    return tensor_img, tensor_l

 # model
def create_model(train_dataset, test_dataset):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),  # 32 Feature-Maps, Größe des Filters, Aktivierungsfunktion, Bildgröße
        tf.keras.layers.MaxPooling2D((2,2)),   # Größe des Pooling-Fensters
        #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        #tf.keras.layers.MaxPooling2D((2,2)),
        #tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'), # Anzahl Neuronen
        #tf.keras.layers.Dense(8),
        #tf.keras.layers.Reshape((4,2), input_shape=(None, 8))
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # train model
    model.fit(train_dataset, epochs=10)
    test_loss, test_accuracy = model.evaluate(test_dataset)
    # TODO: create a folder to save the model. Please do not try to push the model (save it outside the repository)
    # model.save("src/test_model")
    print(f"test loss: {test_loss}")
    print(f"test accuracy: {test_accuracy}")


def load_model(model_path, test_input):
    saved_model = tf.keras.models.load_model(model_path)
    saved_model.fit(test_input)

    print(saved_model)


if __name__ == "__main__":
    tensor_images, tensor_label = get_images(images_path = "./train/" )
    print(f'tensor_image: {tensor_images[420]}')
    print(f'tensor_label: {tensor_label[420]}')

    tf_dataset = tf.data.Dataset.from_tensor_slices((tensor_images, tensor_label))

    # split dataset in train, test & val
    train_size = int(0.7 * 7601) # hart codiert :(
    val_size = int(0.15 * 7601)
    test_size = int(0.15 * 7601)

    full_dataset = tf_dataset.shuffle(20)
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    train_dataset = train_dataset.batch(10)  # Trainingssatz wird in Gruppen von jeweils 10 Elementen aufgeteilt  -> kleinere Menge einfacher für Modell
    test_dataset = test_dataset.batch(10)
    val_dataset = val_dataset.batch(10)

    print(len(test_dataset), len(train_dataset), len(val_dataset))

    create_model(train_dataset=train_dataset, test_dataset=test_dataset)

    # TODO: input path to your saved model
    model_path = "./model/test_model"
    load_model(model_path, test_dataset)


