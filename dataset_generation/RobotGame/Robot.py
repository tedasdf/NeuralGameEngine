
import cv2
import numpy as np
import requests
import time
import serial
import modern_robotics as mr
import math
import matplotlib.pyplot as plt
import os
####
#  Robot class for controlling the robot and capturing images
#  interact with arudnio to send arudino signal for the movement
#  init angle 105,35,0

class Robot:
    def __init__(self,
                 L1, L2, L3, 
                 x_offset , y_offset , z_offset,
                 serial_port='COM3',
                 init_thetas=np.array([90, 90, 45]),
                 test_mode=True):
        
        self.test_mode = test_mode

        # connect to ardunio
        if not self.test_mode:
            # ceonnect to arduino
            self.ser = serial.Serial(serial_port, 9600, timeout=1, write_timeout=1)

        self.num_actions = 7

        # robot parameters
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.init_thetas = init_thetas
        self.theta_range = np.array([[0, 0, 0], [180, 180, 90]])
        
        buffer_x = 2
        buffer_y = 2
        max_reach = L2 + L3
        
        max_xy = max_reach - buffer_x
        max_y = max_reach - buffer_y
    
        self.action_range = np.array([
            [-max_xy , -max_y , L1 - 2] , 
            [max_xy , max_y , L1 + L2 + L3 - 5]
        ])
        print("==================ACTION RANGE========================")
        print(self.action_range)
        print("======================================================")
        self.state = np.zeros((4, 4), dtype=np.float32)  # Homogeneous transformation matrix
        self.thetas = self.init_thetas  # Joint angles in radians
        
        self._init_robot_param(L1, L2, L3)  # Example link lengths
        # state of robot

    def _init_robot_param(self, L1, L2 ,L3):

        # Define joint axes and locations
        w1 = np.array([0, 0, 1])
        q1 = np.array([0, 0, 0])
        v1 = -np.cross(w1, q1)

        w2 = np.array([0, -1, 0])
        q2 = np.array([self.x_offset, 0, L1])
        v2 = -np.cross(w2, q2)

        w3 = np.array([0, 1, 0])
        q3 = np.array([L2+self.x_offset, 0, L1])
        v3 = -np.cross(w3, q3)

        # Screw axes (6x3 matrix)
        S1 = np.hstack((w1, v1))  # shape (6,)
        S2 = np.hstack((w2, v2))
        S3 = np.hstack((w3, v3))
        self.Slist = np.column_stack((S1, S2, S3))  # shape (6,3)

        # Store home configuration
        self.M = np.array([
            [1, 0, 0, L2+self.x_offset],
            [0, 1, 0, 0],
            [0, 0, 1, L1-L3],
            [0, 0, 0, 1]
        ])
        
        self.return_init_pos()

    def return_init_pos(self):
        print(f"{self.thetas}")
        if not self.test_mode:
            success = self.read_state_from_robot()
            if not success:
                raise RuntimeError("Failed to read state from robot.")
                    
        thetas_signal = self.init_thetas
        self.thetas = self.init_thetas
        print(f"Sending signal: {thetas_signal}")
        if not self.test_mode:
            self.ser.write(f"{thetas_signal[0]},{thetas_signal[1]},{thetas_signal[2]}\n".encode())
        
        self.state = self.forward_kinematics(thetas=np.deg2rad(self.thetas))
        print(f"Robot initialized at position: {self.state}")


    def read_state_from_robot(self) -> bool:
        self.ser.write(b"0\n")  # Request current angles from Arduino
        lines = []
        start_time = time.time()
        while len(lines) < 3 and time.time() - start_time < 2.0:  # Timeout after 2s
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode().strip()
                if line:
                    lines.append(line)
        if len(lines) == 3:
            try:
                theta1 = int(lines[0])
                theta2 = int(lines[1])
                theta3 = int(lines[2])
                self.theta = np.array([theta1, theta2, theta3])
                return True
            except ValueError:
                print("Error: Could not parse angles from Arduino.")
        else:
            print("Timeout or incomplete data from Arduino.")
        # Fallback: estimate or return dummy values
        return False  # ← make sure this exists and is valid

    def forward_kinematics(self, thetas):
        return mr.FKinSpace(self.M, self.Slist, thetas)    

    def closest_distance_node(self, points,corresponding_theta, proposed_pos):
        # Compute Euclidean distances
        distances = np.linalg.norm(points - proposed_pos, axis=1)

        # Find closest index
        closest_idx = np.argmin(distances)
        closest_point = points[closest_idx]
        closest_distance = distances[closest_idx]
        closest_theta = corresponding_theta[closest_idx]
        
        return closest_point, closest_theta, closest_distance

    def to_transformed_signal(self,thetas_list, previous_thetas):
        """
        Given a list of theta arrays and a reference (previous_thetas),
        returns the transformed theta (with rounding and +90 on theta3) 
        that is closest to previous_thetas.
        """
        previous_thetas = np.array(previous_thetas)
        min_dist = float('inf')
        best_thetas = None
        print(thetas_list)
        for thetas in thetas_list:
            # Step 1: Transform each theta
            thetas_rounded = np.round(thetas).astype(int)
            thetas_rounded[2] += 90
            thetas_rounded[2] *= -1
            # Step 2: Compare to previous_thetas
            dist = np.linalg.norm(thetas_rounded - previous_thetas)

            if dist < min_dist:
                min_dist = dist
                best_thetas = thetas_rounded

        return best_thetas

    def velocity_sensitivity(self):
        self.J = mr.JacobianSpace(self.Slist, np.deg2rad(self.thetas))
        print("J",self.J)
        dtheta = np.array([0.01745 *2, 0.01745*2, 0.01745*2])  # rad/s

        V = self.J @ dtheta
        omega = V[0:3]  # angular velocity of end-effector (rad/s)
        v = V[3:6]
        return omega, np.abs(v) 
    
    def final_solution(self, action_idx):
        

        print("========================================")
        print("BEFORE")

        print(f"State {self.state}")
        print(f"Position {self.state[0:3,3]}")
        print(f"theta {self.thetas}")


        direction_map = {
            0: np.array([+1, 0, 0]),
            1: np.array([0, +1, 0]),
            2: np.array([0, 0, +1]),
            3: np.array([-1, 0, 0]),
            4: np.array([0, -1, 0]),
            5: np.array([0, 0, -1]),
            6: np.array([0, 0, 0])
        }

        _ , v = self.velocity_sensitivity()
        print("V",v)
        
        proposed_translation = np.multiply(direction_map[action_idx] , v)
        proposed_pos = self.state[0:3, 3] + proposed_translation
        print("proposed pos",proposed_pos)
        try_thetas , success =self.inverse_kinematics_3d( proposed_pos[0], proposed_pos[1], proposed_pos[2])
        if not success:
            return success
        print("result from inverse kinematics:"  ,try_thetas)

        thetas_signal = self.to_transformed_signal(try_thetas, self.thetas)

        print("Thetas signal " ,thetas_signal)
        if not self.test_mode:
            self.ser.write(f"{thetas_signal[0]},{thetas_signal[1]},{thetas_signal[2]}\n".encode())
            self.read_state_from_robot()
        else:
            self.thetas = thetas_signal

        self.state = self.forward_kinematics(np.deg2rad(self.thetas))
        print("=================================================================")
        return success
    

    def move(self, theta):
        delta_theta = np.round(theta).astype(int)
        proposed_thetas = self.thetas + delta_theta

        within_theta_bounds = np.all(proposed_thetas >= self.theta_range[0]) and \
                          np.all(proposed_thetas <= self.theta_range[1])
        if not within_theta_bounds:
            return False
    
        proposed_state = self.forward_kinematics(np.deg2rad(proposed_thetas))

        pos = proposed_state[0:3,3] 

        if not np.all(pos >= self.action_range[0]) and np.all(pos <= self.action_range[1]):
            return False
        
        if not self.test_mode:
             self.ser.write(f"{proposed_thetas[0]},{proposed_thetas[1]},{proposed_thetas[2]}\n".encode())
        
        self.thetas = proposed_thetas
        self.state = proposed_state
        return True

    def step(self, theta ):
        success = self.move(theta = theta)
        return success
    
    def inverse_kinematics_3d(self, x, y, z):
        """
        Calculates inverse kinematics for a 3DOF arm:
        - Joint 1 rotates around Z-axis
        - Joints 2 & 3 act in YZ plane
        Returns the solution closest to previous_thetas if provided.
        """
        x_offset = self.x_offset
        L1,L2,L3 = self.L1, self.L2, self.L3
        # 1. θ1 = base rotation to reach (x, y)
        theta1 = np.arctan2(y, x)  # rotation around Z to reach y-axis

        # 2. Project the target onto the YZ plane by rotating into frame
        y_proj = np.sqrt(x**2 + y**2) - x_offset
        z_eff = z - L1  # offset for base height
        
        # 3. Compute r for planar arm (in YZ)

        r = np.sqrt(y_proj**2 + z_eff**2)

        # Check reachability
        if r > (L2 + L3) or r < abs(L2 - L3):
            return (None, None) ,False

        # 4. Two possible θ3 (elbow angle)
        cos_theta3 = (r**2 - L2**2 - L3**2) / (2 * L2 * L3)
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
        theta3_up =  np.arccos(cos_theta3)
        theta3_down = -theta3_up

        # 5. Corresponding θ2 (shoulder angle)
        k1_up = L2 + L3 * np.cos(theta3_up)
        k2_up = L3 * np.sin(theta3_up)
        theta2_up = np.arctan2(z_eff, y_proj) - np.arctan2(k2_up, k1_up)

        k1_down = L2 + L3 * np.cos(theta3_down)
        k2_down = L3 * np.sin(theta3_down)
        theta2_down = np.arctan2(z_eff, y_proj) - np.arctan2(k2_down, k1_down)

        # Pack solutions
        sol_up = np.array([theta1, theta2_up, theta3_up])
        sol_down = np.array([theta1, theta2_down, theta3_down])

        sol_up_deg = np.rad2deg(sol_up)
        sol_down_deg = np.rad2deg(sol_down)


        return (sol_up_deg, sol_down_deg) , True

 
    def reset(self):
        # Initialize the robot's position to the initial position
        self.return_init_pos()
        # self.capture_led_sequence()

 

if __name__ == "__main__":

    robot = Robot(L1=7.14, L2=7, L3=8.34 , x_offset=0.9185 , y_offset=0, z_offset=0, serial_port='COM3', test_mode=True)
    import keyboard
    import time
    import matplotlib.pyplot as plt

    try:
        print("Type in desired joint angles (theta1, theta2, theta3) in degrees.")
        print("Example: 90 45 30   |   Type 'exit' to quit.")

        while True:
            user_input = input("Enter new thetas (°): ")

            if user_input.lower() == "exit":
                print("Exiting...")
                break

            try:
                thetas = np.array([float(x) for x in user_input.strip().split()])
                if thetas.shape[0] != 3:
                    print("Please enter exactly three values.")
                    continue
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces.")
                continue

            

            success = robot.move(thetas)
            if success:
                print("Move successful. New position:", robot.state[0:3, 3])
                print("Current joint angles (°):", robot.thetas)
            else:
                print("Move rejected (joint or position limit).")

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")
