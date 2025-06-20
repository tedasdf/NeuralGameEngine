
import cv2
import numpy as np
import requests
import time
import serial




####
#  Robot class for controlling the robot and capturing images
#  interact with arudnio to send arudino signal for the movement
#  
# 
class Robot:
    def __init__(self, 
                 frame=None,
                 rl_model=None,
                 init_position=(0, 0), 
                 num_colors=5):
        self.url = "http://192.168.0.57/capture"
        self.state = None
        self.init_position = init_position
        self.ser = serial.Serial('COM3', 9600, timeout=1, write_timeout=1)
        self.frame = frame


    def calculated_screw_axis(self, theta1, theta2, theta3):
        # Placeholder for calculating the screw axis based on joint angles
        # This should return the screw axis for the robot's joints
        screw_axis = np.array([theta1, theta2, theta3])
        return screw_axis
    


    def forward_kinematics(self, theta1, theta2, theta3):
        # Placeholder for forward kinematics calculation
        # This should return the position of the robot's end effector based on joint angles
        x = theta1 + theta2 + theta3    
    
    def backward_kinematics(self, x, y):
        # Placeholder for backward kinematics calculation
        # This should return the joint angles based on the end effector position
        theta1 = x / 3
        theta2 = y / 3
        theta3 = (x + y) / 6
        return theta1, theta2, theta3
    

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
        self.state += action
        theta1, theta2, theta3 = self.backward_kinematics(self.state)
        self.ser.write(f"{theta1},{theta2},{theta3}\n".encode())


        # Send the angles to the robot via serial communication
        self.estimated_state = self.forward_kinematics(theta1, theta2, theta3)
        self.actual_sate = self.read_state_from_robot()


        # This could involve sending commands to the robot's motors
        print(f"Moving robot with action: {action}")

    def step(self, action ):
        # capture image 
        # neural network 
        # move
        # new state
        self.move(action)
        obs = self.camera_cap()
        if obs is not None:
            cv2.imshow("Robot Camera", obs)
            cv2.waitKey(1)

        return obs, self.estimated_state, self.actual_sate
        


if __name__ == "__main__":
    robot = Robot()
    while True:
        action = np.random.rand(3)  # Random action for testing
        obs, estimated_state, actual_state = robot.step(action)
        print(f"Estimated State: {estimated_state}, Actual State: {actual_state}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    robot.ser.close()