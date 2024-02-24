import tensorflow as tf
import os
import json

def get_images():

    images_path = "./train/"   # path
    images_dirs = os.listdir(images_path)   # folder of single buchstabe
    tensor_images = []   

    # get images as tensors
    for img in images_dirs:   # geht jedes Bild in Ordner durch
        filepath = os.path.join(images_path, img)

        if filepath.lower().endswith(('.jpeg', '.jpg')):
            image = tf.io.read_file(filepath)
            tensor_image = tf.io.decode_image(image, channels=3, dtype=tf.dtypes.float32)
            tensor_image = tf.image.resize(tensor_image, [256, 256])
            tensor_images.append(tensor_image)
    
    return tensor_images


def get_label():

    label_path = os.path.join(os.path.join(os.getcwd(), "training"), "label_+_augm_label")
    label_dirs = os.listdir(label_path)
    tensor_label = []

    # get labels als tensors
    for label in label_dirs:
        filepath = os.path.join(label_path, label)

        if filepath.lower().endswith(('.json')):
            label_data = json.load(open(filepath))
            label_points = label_data["shapes"][0]["points"]
            tensor_label.append(label_points)

    return tensor_label

if __name__ == "__main__":
    get_images()
    get_label()



"""
    tf_dataset = tf.data.Dataset.from_tensor_slices((get_images(), get_label()))
    #tf_dataset = tf_dataset.batch(10)

    # split dataset in train, test & val
    train_size = int(0.7 * 3906) # hart codiert :(
    val_size = int(0.15 * 3906)
    test_size = int(0.15 * 3906)

    full_dataset = tf_dataset.shuffle(20)
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    train_dataset = train_dataset.batch(10)
    test_dataset = test_dataset.batch(10)
    val_dataset = val_dataset.batch(10)

    print(len(test_dataset), len(train_dataset), len(val_dataset))

    # model
    def create_model(train_dataset, test_dataset):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300,200,3)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(8),
            tf.keras.layers.Reshape((4,2), input_shape=(None, 8))
        ])

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        # train model
        model.fit(train_dataset, epochs=10)
        test_loss, test_accuracy = model.evaluate(test_dataset)
        # TODO: create a folder to save the model. Please do not try to push the model (save it outside the repository)
        # model.save("src/test_model")
        print(f"test loss: {test_loss}")
        print(f"test accuracy: {test_accuracy}")

    #create_model(train_dataset, test_dataset)

    def load_model(model_path, test_input):
        saved_model = tf.keras.models.load_model(model_path)
        saved_model.fit(test_input)

        print(saved_model)

"""

# TODO: input path to your saved model
#model_path = "/Users/julia/Desktop/Uni/Semester_4/Anwendungsprojekt_Informatik/test_model"
#load_model(model_path, test_dataset)
