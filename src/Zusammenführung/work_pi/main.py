import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QDialog, QDialogButtonBox, QVBoxLayout, QPushButton, \
    QLineEdit, QTextEdit, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from picamera2 import Picamera2, Preview
from libcamera import Transform

from camera_gui import Ui_Dialog  # Generierte UI-Klasse

import os

from natsort import natsorted

from hand_image_capture import HandImageCapture
from get_predictions import Model, ImageClassificationBase, NaturalSceneClassification

from shape import Shape


class InputDialog(QDialog):
    """
    Klasse InputDialog stellt die GUI zur Eingabe des Buchstabens nach der Aufnahme eines Bildes.
    Nachdem die Klasse instanziiert wurde, wird ein Fenster geöffnet, in dem der Benutzer ein
    Buchstabe eingeben kann und die Eingabe abschicken kann.
    """

    def __init__(self):
        """Instanziieren der Klasse. Die GUI wird erstellt und die Buttons werden den Events
        zugeordnet. Wenn ein Button gedrückt wird, wird ein Event ausgelöst, welches
        dann abgefangen werden kann."""
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
        """Gibt die Eingabe des Benutzers zurück."""
        return self.input_field.text()


class OutputDialog(QDialog):
    """
    Klasse OutputDialog stellt die GUI zur Ausgabe von Nachrichten bereit.
    Nachdem die Klasse instanziiert wurde, kann ein Text in das Ausgabefeld gesetzt werden.
    Mithilfe der Funktion .show() kann die GUI dann angezeigt werden.
    """

    def __init__(self):
        """Instanziieren der Klasse. Die GUI wird erstellt und der Close-Button wird dem Event
        'rejected' zugeordnet. Wenn der Button gedrückt wird, wird ein Event ausgelöst, welches
        dann abgefangen werden kann."""
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
        """Fügt einen übergebenen Text dem Ausgabefeld hinzu.
        :param text: Text, welcher dem Feld hinzugefügt werden soll."""
        self.console_output.append(text)


class MyApp(QMainWindow):
    """Main-Klasse, welches die GUI und die Logik enthält."""

    def __init__(self):
        """Instanziieren der Klasse und Konfiguration der Kamera.
        Außerdem wird hier ein QTimer gesetzt, mit welchem eine ständige Aktualisierung möglich ist (Live-Bild)"""
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Picamera2 setup
        self.picam2 = Picamera2()
        #self.picam2.configure(self.picam2.create_video_configuration(main={"format": 'RGB888', "size": (640, 480)},
         #                                                            transform=Transform(hflip=1, vflip=1)))
        self.picam2.configure(self.picam2.create_video_configuration(main={"format": 'RGB888', "size": (640, 480)}))
        self.picam2.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)  # hiermit wird eine Schleife gestartet
        self.timer.start(30)  # 30 ms für ungefähr 33 FPS

        self.ui.btnCaptureImage.clicked.connect(self.capture_image)

        self.ui.btnKI.clicked.connect(self.run_ki)

        self.ui.btnQuit.clicked.connect(self.close_event)

        self.output_text = ""

        self.hand_image_capture = HandImageCapture()

        self.shape_manager = Shape()
        #self.shape_manager.circle()
        #self.shape_manager.lower_pen()
        #self.shape_manager.lift_pen()

    def capture_image(self):
        """Wird durch den Button 'Bild aufnehmen' aufgerufen. Hier wird das aktuelle Bild der Kamera
        aufgenommen. Der Timer wird gestoppt, wodurch das Live-Bild in der GUI gestoppt wird.
        Außerdem wird das Eingabefeld (Klasse InputDialog) geöffnet, in dem der Benutzer
        den Buchstaben eingeben kann. Das Bild wird dabei gespeichert.
        Zuletzt wird der Timer wieder gestartet, wodurch sich das Live-Bild wieder aktualisiert."""
        # print("Bild aufgenommen")
        im = self.picam2.capture_array()
        self.timer.stop()
        input_dialog = InputDialog()
        description = ""
        if input_dialog.exec_() == QDialog.Accepted:
            description = input_dialog.get_input()
            # print(f"Benutzer hat eingeben: {description}")
            # self.outputDialog.append_text("Das ist ein Test")
            # self.outputDialog.show()

        # Speichere das Bild mit der Beschreibung
        self.hand_image_capture.save_image(im, description)

        self.timer.start()

    def get_word(self, data_dir):
        """Anhand der Dateinamen wird das tatsächliche Wort ermittelt. Da diese den Aufbau
        NUMMER_BUCHSTABE.jpg haben, kann der Buchstabe extrahiert werden.
        :param data_dir: Pfad zu den zugeschnittenen Bildern
        :return: Gibt ein Array mit den Buchstaben zurück."""
        letters_per_img = []
        for filename in natsorted(os.listdir(data_dir)):
            if filename.endswith('.jpg'):
                letter = filename.split('_')[1].split('.')[0]
                letters_per_img.append(letter)
            else:
                print("Keine gültige Datei.")
        return letters_per_img

    def check_prediction(self, real_letters, predicted_letters):
        """Überprüft, ob das vorhergesagte Wort mit dem tatsächlichen Wort übereinstimmt.
        :param real_letters: Buchstaben, welche durch den Benutzer eingegeben wurden.
        :param predicted_letters: Buchstaben, welche durch das Model vorhergesagt wurden.
        :return: Gibt das tatsächliche Wort zurück."""
        predicted_word = ''.join(predicted_letters)
        real_word = ''.join(real_letters)
        self.output_text = f"Vorhergesagtes Wort: {predicted_word}\n"
        self.output_text += f"Tatsächliches Wort: {real_word}\n"
        print(f'vorhergesagtes Wort: {predicted_word}')
        print(f'tatsächliches Wort: {real_word}')

        if predicted_word == real_word:
            self.output_text += f"Die Worte stimmen überein.\n"
        else:
            self.output_text += f"Die Worte stimmen nicht überein.\n"
            differing_letters = []

            for i in range(len(real_letters)):
                if real_letters[i] != predicted_letters[i]:
                    differing_letters.append((real_letters[i], predicted_letters[i]))

                # if differing_letters:
                # print("Die Wort stimmen nicht überein. Folgende Buchstaben wurden falsch erkannt: ")
                # print(differing_letters)
                # for letter_pair in (len(differing_letters/2)):
                #    print(f'statt {letter_pair[0]} wurde folgender Buchstabe
                #    fälschlicherweise vorhergesagt: {letter_pair[1]}')
        return real_word

    def draw_shape(self, x):
        """Zeichnen der Form."""
        x = x.lower()
        if x != "kreis" and x != "rechteck" and x != "dreieck" and x != "herz"\
                and x != "linie":
            self.output_text += "Keine gültige Form übergeben. Überspringe das Zeichnen..."
        else:
            self.output_text += f"Es wird nun ein {x} gezeichnet."

        print("DEBUG: ", self.output_text)
        self.timer.stop()

        output_dialog = OutputDialog()
        output_dialog.append_text(self.output_text)
        if output_dialog.exec_() == QDialog.Rejected:
            self.resetValues()
            self.timer.start()

        if x == "kreis":
            self.shape_manager.circle()
        # elif x == "rechteck":
            # self.shape_manager.rectangle()
        # elif x == "dreieck":
            # self.shape_manager.rectangle()
        # elif x == "herz":
            # self.shape_manager.heart()
        elif x == "linie":
            self.shape_manager.line()
        else:
            print("Leider keine gültige Form...")

    def resetValues(self):
        """Setzt die Werte der Klasse Camera() und die Variable für den Ausgabetext zurück. Diese
        Funktion wird aufgerufen, nachdem der Roboter die Form fertig gezeichnet hat."""
        self.hand_image_capture.reset_values()
        self.output_text = ""

    def run_ki(self):
        """Wird durch den Button 'Wort erkennen' aufgerufen. Hier werden die Funktionen aufgerufen,
        um die Hände in den Bildern zu erkennen und diese zuzuschneiden. Außerdem werden die in
        einem neuen Ordner gespeichert.
        Zuletzt wird das Model geladen, welches dann eine Vorhersage trifft. Es findet
        dabei eine Überprüfung statt, ob die Vorhersage mit dem tatsächlichen Wort
        übereinstimmt-"""
        new_path = self.hand_image_capture.get_hand()
        model_path = '/home/pi/Desktop/Working_Directory/work_pi/models/model_new_epoch10.pth'
        data_dir = new_path
        imageClassificationBase = ImageClassificationBase()
        naturalSceneClassification = NaturalSceneClassification()
        model = Model(model_path, data_dir)
        predicted_letters = model.predict_image()

        # get correct word
        real_letters_per_img = self.get_word(data_dir)
        word = self.check_prediction(real_letters_per_img, predicted_letters)
        self.draw_shape(word)

    def update_frame(self):
        """Mit jedem Frame wird diese Methode aufgerufen. Dabei wird das Live-Bild
        aktualisiert.
        Es wird das aktuelle Bild der Kamera gespeichert und zu einem QImage formatiert.
        Dieses kann dann im Label angezeigt werden."""
        frame = self.picam2.capture_array()
        # Convert the frame to QImage
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.ui.lblLiveView.setPixmap(QPixmap.fromImage(convert_to_qt_format))
        self.ui.lblLiveView.setFixedSize(w, h)

    def close_event(self, event):
        """Wird aufgerufen, wenn das Fenster geschlossen wird.
        Kann ebenfalls durch einen Button aufgerufen werden, wodurch das Programm
        beendet wird."""
        self.picam2.stop()
        super().close_event(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())

