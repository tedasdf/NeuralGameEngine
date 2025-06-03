
import cv2
import numpy as np
import requests
import time
import serial

class Robot:
    def __init__(self, 
                 rl_model=None,
                 init_position=(0, 0), 
                 num_colors=5):
        self.url = "http://192.168.0.57/capture"
        self.state = None
        self.init_position = init_position
        self.ser = serial.Serial('COM3', 9600, timeout=1, write_timeout=1)

    def camera_cap(self):
        try:
            response = requests.get(self.url, stream=True, timeout=5)
            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
                else:
                    print("Failed to decode image")
                    return None
            else:
                print(f"Error: Status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception during image capture: {e}")
            return None

    def move(self, action):
        # Placeholder for movement logic
        # find the angle and send to ardunio
        self.ser.write(f"{theta1},{theta2},{theta3}\n".encode())
        # This could involve sending commands to the robot's motors
        print(f"Moving robot with action: {action}")

    def update_state(self, new_state):
        self.state = new_state

    def step(self):
        # capture image 
        # neural network 
        # move
        # new state

        


