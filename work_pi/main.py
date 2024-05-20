import os
from get_images import Camera
from get_predictions import Model
from get_predictions import ImageClassificationBase
from get_predictions import NaturalSceneClassification

def get_word(data_dir):
    # takes the label from file name of each image and saves it in array
    letters_per_img = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            letter = filename.split('_')[1].split('.')[0]
            letters_per_img.append(letter)
        else:
            print("Keine gültige Datei.")
    return letters_per_img

def check_prediction(real_letters, predicted_letters):
    # checks wether predicted word and real word are the same
    predicted_word = ''.join(predicted_letters)
    real_word = ''.join(real_letters)
    print(f'vorhergesagtes Wort: {predicted_word}')
    print(f'tatsächliches Wort: {real_word}')

    if predicted_word == real_word:
        print("Die Worte stimmen überein.")
    else:
        differing_letters = []

        for i in range(len(real_letters)):
            if real_letters[i] != predicted_letters[i]:
                differing_letters.append((real_letters[i], predicted_letters[i]))
            
            if differing_letters:
                print("Die Wort stimmen nicht überein. Folgende Buchstaben wurden falsch erkannt:")
                for letter_pair in differing_letters:
                    print(f'statt {letter_pair[0]} wurde folgender Buchstabe fälschlicherweise vorhergesagt: {letter_pair[1]}')
    return real_word


def correct_word(x):
    return x

if __name__ == '__main__':
    camera = Camera()
    camera.take_image()

    # get predictions
    model_path = './cnn/cnn2/versuch2/output/model_epoch15.pth'
    data_dir = input("Gib den Pfad für die vorherzusagenden Bilder an: ")
    imageClassificationBase = ImageClassificationBase()
    naturalSceneClassification = NaturalSceneClassification()
    model = Model(model_path, data_dir)
    predicted_letters = model.predict_image()
    
    # get correct word
    real_letters_per_img = get_word(data_dir)
    word = check_prediction(real_letters_per_img, predicted_letters)
    real_word_for_roboter = correct_word(word)