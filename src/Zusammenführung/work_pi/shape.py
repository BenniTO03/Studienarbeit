import nxt.locator
import nxt
from nxt.motor import *
import time


class Shape:
    """Klasse Shape enthält alle Formen, die der Roboter zeichnen kann.
    Es reicht, wenn die Klasse einmal instanziiert wird und dann nur die
    entsprechenden Funktionen aufgerufen werden."""

    def __init__(self):
        """Instanziieren der Klasse und setzen der Variablen."""
        self.name = None
        self.WHEEL_DIAMETER = 5.6
        self.TRACK_WIDTH = 12.0

        self.power = 75

        self.b = None
        self.motor_right = None
        self.motor_left = None
        self.pen_motor = None

        self.find_brick_and_get_motors()

    def find_brick_and_get_motors(self):
        """Sucht nach dem Brick und instanziiert die Motoren"""
        self.b = nxt.locator.find(host="00:16:53:0C:63:A6")
        self.motor_left = self.b.get_motor(nxt.motor.Port.A)
        self.motor_right = self.b.get_motor(nxt.motor.Port.B)
        self.pen_motor = self.b.get_motor(nxt.motor.Port.C)

    def degree_to_rotation(self, degree):
        """Übersetzt die Gradzahl in die Anzahl an Umdrehungen, die der Motor drehen muss."""
        return (degree * self.TRACK_WIDTH) / (self.WHEEL_DIAMETER * 360)

    def turn_degrees(self, degrees):
        """Dreht den Roboter.
        :param degrees: Gradzahl, um die der Roboter gedreht werden soll."""
        turn_rotations = self.degree_to_rotation(degrees)
        tacho_limit = turn_rotations * 3.14159 * self.WHEEL_DIAMETER / 360

        self.motor_left.run(self.power)
        self.motor_right.run(-self.power)

        time.sleep(tacho_limit / abs(self.power) * 2.5)

        self.motor_left.brake()
        self.motor_right.brake()

    def circle(self, radius=5):
        """Zeichnen einen Kreis.
        :param radius: Radius des Kreises. Standardmäßig sind 5 cm eingestellt."""
        # Umfang des Kreises
        circumference = 2 * 3.14159 * radius

        # Geschwindigkeit der Räder
        left_wheel_speed = self.power
        right_wheel_speed = self.power * (
                    (radius - (self.TRACK_WIDTH / 2)) / (radius + (self.TRACK_WIDTH / 2)))

        # Zeit berechnen, um den Kreis zu zeichnen
        rotations_needed = circumference / (3.14159 * self.WHEEL_DIAMETER)
        time_to_complete_circle = (rotations_needed * 360) / left_wheel_speed  # Zeit in Sekunden

        self.lower_pen()
        # Start der Motoren
        self.motor_right.turn(100, 4*360)
        #self.motor_right.run(int(right_wheel_speed))

        # Kreis zeichnen für die berechnete Zeit
        #time.sleep(time_to_complete_circle)

        # Motoren stoppen
        self.motor_left.brake()
        self.motor_right.brake()
        self.lift_pen()

    def rectangle(self, length=10, width=5):
        """Zeichnen eines Rechtecks."""
        for _ in range(2):
            # Senke den Stift
            self.lower_pen()

            # Gerade Strecke - Länge
            self.line(length)

            # Hebe den Stift
            self.lift_pen()

            # 90-Grad Kurve
            self.turn_degrees(90)

            # Senke den Stift
            self.lower_pen()

            # Gerade Strecke - Breite
            self.line(width)

            # Hebe den Stift
            self.lift_pen()

            # 90-Grad Kurve
            self.turn_degrees(90)

    def triangle(self, side_length=10):
        """Zeichnen eines Dreiecks.
        :param side_length: Länge der Seiten. Standardmäßig sind 10 cm eingestellt."""
        for _ in range(3):
            # Senke den Stift
            self.lower_pen()

            # Gerade Strecke - Seite des Dreiecks
            self.line(side_length)

            # Hebe den Stift
            self.lift_pen()

            # 120-Grad Kurve
            self.turn_degrees(120)

    def heart(self, straight_length=10, curve_length=5):
        """Zeichnen eines Herzes.
        :param straight_length: Länge
        :param curve_length: Kurvenlänge"""
        # Zeichne das erste halbe Herz (rechts oben)
        self.lower_pen()
        self.line(straight_length)
        self.turn_degrees(45)
        self.line(curve_length)
        self.turn_degrees(90)
        self.line(curve_length)
        self.turn_degrees(45)
        self.line(straight_length)

        # Hebe den Stift
        self.lift_pen()

        # Drehe den Roboter um 180 Grad für die andere Hälfte des Herzens
        self.turn_degrees(180)

        # Zeichne das zweite halbe Herz (links oben)
        self.lower_pen()
        self.line(straight_length)
        self.turn_degrees(-45)
        self.line(curve_length)
        self.turn_degrees(-90)
        self.line(curve_length)
        self.turn_degrees(-45)
        self.line(straight_length)

        # Hebe den Stift
        self.lift_pen()

    def line(self, distance_cm=5):
        """Zeichnet eine gerade Linie.
        :param distance_cm: Länge der Linie. Standardmäßig sind 5 cm eingestellt."""
        rotations = (distance_cm * 360) / (3.14159 * self.WHEEL_DIAMETER)
        tacho_limit = rotations * 3.14159 * self.WHEEL_DIAMETER / 360
        self.lower_pen()
        self.motor_left = self.b.get_motor(nxt.motor.Port.A)
        self.motor_right = self.b.get_motor(nxt.motor.Port.B)

        self.motor_left.run(self.power)
        self.motor_right.run(self.power)

        time.sleep(2)

        self.motor_left.brake()
        self.motor_right.brake()
        
        self.lift_pen()

    def lower_pen(self):
        """Setzt den Stift ab, in dem der Motor C gedreht wird."""
        self.pen_motor = self.b.get_motor(nxt.motor.Port.C)
        self.pen_motor.turn(50, 70) # Stift senken
        print("Stift senken")

    def lift_pen(self):
        """Hebt den Stift an, in dem der Motor C gedreht wird."""
        self.pen_motor = self.b.get_motor(nxt.motor.Port.C)
        self.pen_motor.turn(-50, 70) # Stift hoch
        print("Stift hoch machen")
