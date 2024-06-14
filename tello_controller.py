"""
This is a class that performs Tello drone controlling

Main functions:
check_battery: if battery less than 5%, reject takeoff command; if less than 5% on the air, perform land command
takeoff: if battery less than 15%, give an alert. If not on the air, take off.
move_forward, move_back, move_left, move_right: move to the corresponding direction for "distance" cm. (Min: 20 cm)
land: perform land command if on the air
emergency_land: perform land command immediately
emergency: tune off the motor
check_flying_status: check if the drone on the air or not
test_drone: test if every method is working
"""
from djitellopy import Tello
import time
import logging
import threading

class TelloController:

    def __init__(self, debug=False):
        if not debug:
            self.setup()

    def setup(self):
        logging.basicConfig(level=logging.INFO)
        # Create Tello object and connect
        self.tello = Tello()
        self.tello.connect()
        # Check battery - if lower than 5% will not perform commands (need to be implemented)
        self.check_battery()
        self.is_flying = False  # whether the drone is flying or not
        self.stabilization_time = 0 # Time in seconds to stabilise the drone after actions
    def check_battery(self):
        # Check battery level
        battery = self.tello.get_battery()
        logging.info(f"Battery level: {battery}%")
        return battery



    def takeoff(self):
        if self.check_battery() > 5 and not self.is_flying:  # Execute take-off only when the drone is not in the air
            if self.check_battery() < 15:
                logging.info("Battery will ran out, please land soon")
            self.tello.takeoff()
            self.is_flying = True
            logging.info("Drone has taken off")
            #time.sleep(self.stabilization_time)

    def start_height_monitoring(self, max_height=100):
        # Create a thread to perform height monitoring
        height_thread = threading.Thread(target=self.check_height, args=(max_height,))
        height_thread.start()

    def check_height(self, max_height = 100):
        buffer = 20
        try:
            # Continuously monitor the drone's height and prevent it from exceeding the max height
            while True:
                height = self.tello.get_height()
                logging.info(f"Current height: {height} cm")
                if height < max_height - buffer:
                    self.tello.send_rc_control(0, 0, 20, 0)  # Continue ascending

                elif height >= max_height + buffer:  # Give some buffer to stop before reaching the limit
                    self.tello.send_rc_control(0, 0, -20, 0)
                if self.check_battery() < 5:
                    self.land()
                    logging.info("Battery run out, please change the battery")
                #time.sleep(0.1)  # Sleep for 100 milliseconds
        except Exception as e:
            logging.error(f"Check height failed: {e}")
            self.tello.land()

    def move_up(self):
        self.tello.send_rc_control(0, 0, 20, 0)  # Continue ascending
        #self.tello.set_speed()
    def move_down(self):
        self.tello.send_rc_control(0, 0, -20, 0)

    def move_forward(self, distance):
        try:
            self.tello.move_forward(distance)
            logging.info(f"Moved forward {distance} cm")
            time.sleep(self.stabilization_time)  # Stabilize after movement
        except Exception as e:
            logging.error(f"Move forward failed: {e}")

    def move_left(self, distance):
        try:
            self.tello.move_left(distance)
            logging.info(f"Moved left {distance} cm")
            time.sleep(self.stabilization_time)
        except Exception as e:
            logging.error(f"Move left failed: {e}")

    def move_right(self, distance):
        try:
            self.tello.move_right(distance)
            logging.info(f"Moved right {distance} cm")
            time.sleep(self.stabilization_time)
        except Exception as e:
            logging.error(f"Move right failed: {e}")

    def move_back(self, distance):
        try:
            self.tello.move_back(distance)
            logging.info(f"Moved back {distance} cm")
            time.sleep(self.stabilization_time)
        except Exception as e:
            logging.error(f"Move back failed: {e}")

    def rotate(self, degree):
        self.tello.rotate_clockwise(degree)

    def flip_left(self):
        self.tello.flip('l')

    def flip_right(self):
        self.tello.flip('r')

    def flip_forward(self):
        self.tello.flip('f')

    def flip_backward(self):
        self.tello.flip('b')

    def land(self):
        if self.is_flying:  # Perform landings only when the drone has already taken off
            self.tello.land()
            self.is_flying = False
            logging.info("Drone has landed")
            time.sleep(self.stabilization_time)  # Stabilising drones
            self.tello.end()
            logging.info("Disconnected from Tello drone")

    def emergency_land(self):
        self.tello.land()
        self.is_flying = False
        logging.info("Drone has landed")
        self.tello.end()
        logging.info("Disconnected from Tello drone")

    def emergency(self):
        self.tello.emergency()

    def check_flying_status(self):
        return self.is_flying

    def preset_command(self):
        self.move_forward(100) # Min is 20cm
        self.rotate(90)
        self.move_forward(20)
        self.rotate(90)
        self.move_forward(100)
        self.rotate(90)
        self.move_forward(20)
        self.rotate(90)

    def preset_command_2(self):
        self.flip_right()
        self.flip_left()


