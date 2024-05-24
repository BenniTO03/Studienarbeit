import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QDialog, QDialogButtonBox, QVBoxLayout, QPushButton, QLineEdit, QTextEdit, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from picamera2 import Picamera2, Preview
from libcamera import controls
from libcamera import Transform
from camera_gui import Ui_Dialog  # Ersetze durch den Namen der generierten UI-Python-Datei

import os

from natsort import natsorted

import time

from get_images_edited import Camera
from get_predictions import Model
from get_predictions import ImageClassificationBase
from get_predictions import NaturalSceneClassification

class InputDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eingabe Dialog")

        # Dialog Layout
        layout = QVBoxLayout()

        # Eingabefeld
        self.input_field = QLineEdit(self)
        layout.addWidget(self.input_field)

        # OK und Cancel Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

        # Signale verbinden
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_input(self):
        return self.input_field.text()

class OutputDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ausgabe")

        # Dialog Layout
        layout = QVBoxLayout()

        # TextEdit for console output
        self.console_output = QTextEdit(self)
        self.console_output.setReadOnly(True)
        layout.addWidget(self.console_output)
        
        # OK und Cancel Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        
        self.setLayout(layout)

    def append_text(self, text):
        self.console_output.append(text)

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Picamera2 setup
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(main={"format": 'RGB888', "size": (640, 480)},transform=Transform(hflip=1, vflip=1)))
        self.picam2.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame) # hier mit wird eine Schleife gestartet
        self.timer.start(30)  # 30 ms für ungefähr 33 FPS
        
        self.ui.btnCaptureImage.clicked.connect(self.captureImage)
        
        self.ui.btnKI.clicked.connect(self.run_ki)
        
        self.ui.btnQuit.clicked.connect(self.closeEvent)
        
        self.output_text = ""
        
        self.get_images = Camera()  
    
    # Wird mit dem Button "Bild aufnehmen" aufgerufen: nimmt das aktuelle Bild auf, Stoppt die Aktualisierung des Live-Bildes
    # öffnet ein Eingabegeld für den echten Buchstaben und speichert das Bild
    # Zuletzt wird das Live-Bild wieder aktiviert
    def captureImage(self):
            # print("Bild aufgenommen")
            im = self.picam2.capture_array()
            self.timer.stop()
            #self.picam2.stop_preview()
            #self.picam2.stop()
            input_dialog = InputDialog()
            if input_dialog.exec_() == QDialog.Accepted:
                    description = input_dialog.get_input()
                    # print(f"Benutzer hat eingeben: {description}")          
                    # self.outputDialog.append_text("Das ist ein Test")
                    # self.outputDialog.show()
        
            # Speichere das Biild mit der Beschreibung     
            self.get_images.save_image(im, description)
            
            self.timer.start()
    
    # ?
    def get_word(self, data_dir):
            # takes the label from file name of each image and saves it in array
            letters_per_img = []
            for filename in natsorted(os.listdir(data_dir)):
                if filename.endswith('.jpg'):
                    letter = filename.split('_')[1].split('.')[0]
                    letters_per_img.append(letter)
                else:
                    print("Keine gültige Datei.")
            return letters_per_img

    # Überprüft, ob die Vorhersage mit dem echten Wort übereinstimmt
    def check_prediction(self, real_letters, predicted_letters):
            # checks wether predicted word and real word are the same
            predicted_word = ''.join(predicted_letters)
            real_word = ''.join(real_letters)
            self.output_text = f"Vorhergesagtes Wort: {predicted_word}\n"
            self.output_text += f"Tatsächliches Wort: {real_word}\n"
            # print(f'vorhergesagtes Wort: {predicted_word}')
            # print(f'tatsächliches Wort: {real_word}')

            if predicted_word == real_word:
                self.output_text += f"Die Worte stimmen überein.\n"
            else:
                differing_letters = []

                for i in range(len(real_letters)):
                    if real_letters[i] != predicted_letters[i]:
                        differing_letters.append((real_letters[i], predicted_letters[i]))
                    
                    #if differing_letters:
                        #print("Die Wort stimmen nicht überein. Folgende Buchstaben wurden falsch erkannt:")
                        #print(differing_letters)
                        #for letter_pair in (len(differing_letters/2)):
                        #    print(f'statt {letter_pair[0]} wurde folgender Buchstabe fälschlicherweise vorhergesagt: {letter_pair[1]}')
            return real_word

    # Hier Funktion aufrufen, damit der Roboter zeichnen.
    def correct_word(self, x):
            #if x == "Kreis":
            #    Kreis()
            self.output_text += f"Es wird nun ein {x} gezeichnet."
            print(self.output_text)
            self.timer.stop()
            output_dialog = OutputDialog()
            output_dialog.append_text(self.output_text)
            if output_dialog.exec_() == QDialog.Rejected:
                time.sleep(5)
                print(f"Fertig gezeichnet. Bereit für neue Form")
                self.resetValues()
                self.timer.start()
         
    # Setzt die Werte der Klasse Camera() zurück, damit ein weiterer Durchlauf gestartet wird
    def resetValues(self):
            self.get_images.reset_values()
            self.output_text = ""
      
    # Wird mit dem Button "KI" aufgerufen und startet: Hände in den Bildern erkennen, Vorhersage treffen
    def run_ki(self):
            new_path = self.get_images.get_hand()
            model_path = '/home/pi/Desktop/Working_Directory/work_pi/model_epoch15.pth'
            data_dir = new_path
            imageClassificationBase = ImageClassificationBase()
            naturalSceneClassification = NaturalSceneClassification()
            model = Model(model_path, data_dir)
            predicted_letters = model.predict_image()
    
            # get correct word
            real_letters_per_img = self.get_word(data_dir)
            word = self.check_prediction(real_letters_per_img, predicted_letters)
            real_word_for_roboter = self.correct_word(word)
    
    # Mit jedem Frame wird diese Methode aufgerufen: Aktualisierung des Live-Bidles   
    def update_frame(self):
        frame = self.picam2.capture_array()
        # Convert the frame to QImage
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.ui.lblLiveView.setPixmap(QPixmap.fromImage(convert_to_Qt_format))
        self.ui.lblLiveView.setFixedSize(w, h)

    # Wird aufgerufen, wenn das Fenster geschlossen wird
    def closeEvent(self, event):
        self.picam2.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
