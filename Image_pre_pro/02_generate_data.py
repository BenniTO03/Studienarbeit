"""
Datei zum Erstellen neuer Bilder

einstellbar:
  - cap = cv2.VideoCapture(0)  -> definiert Kamera
  - dataset_size = Anzahl Bilder
  - number_of_classes = Anzahl verschiedener Zeichen
"""

import os
import cv2

# Ordner in den gespeichert werden soll
DATA_DIR = './sep1'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 2
dataset_size = 50

# Kamera
desired_fps = 10  # Bilder pro Sekunde
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, desired_fps)
# geht sequentiell alle angegebenen Klassen durch 
# nennt jeden Ordner als Zahl
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    """
    öffnet separates Fenster mit Live-Kamera
    """
    while True:
        ret, frame = cap.read()
        # Position des Textes, Schriftgröße, Schriftfarbe, Linienstärke 
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # macht Bilder von counter bis dataset_size
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(100)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()