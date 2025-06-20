import gymnasium as gym
from gymnasium import spaces
import serial
import numpy as np
import time

class RobotEnv(gym.Env):
    
    def __init__(
            self,
            robot_model,
            num_colors,
            num_lights,
            time_limit,
            serial_port,
            baud_rate = 9600,
            timeout = 1,
        ):
        super(RobotEnv, self).__init__()
        # Define action and observation space
        # the movemnet of the robot
        self.num_actions = self.robot_model.num_actions
        self.action_space = spaces.MultiBinary(self.num_actions)

        # Observation space ( 0 : no light , 1 : red , 2 : green , 3 : blue)
        self.num_colors = num_colors 
        self.num_lights = num_lights
        low , high = self.robot_model.get_range()
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=low, high=high, dtype=np.float32),
            "lights": spaces.MultiDiscrete([self.num_colors] * self.num_lights)
        })


        self.current_sequence = None
        self.time_limit = time_limit
        self.robot_model = robot_model
        self.current_time = None

        self.reward = None
        self.steps = None
        
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_position = self.robot_model.init_position()
        self.current_sequence = [np.random.randint(1, self.num_colors) for _ in range(self.num_lights)]
        self.steps = 0
        self.current_time = time.time()
        self.reward = 0.0
        # send the current sequence to hardware


        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "position": np.array(self.robot_position, dtype=np.int32),
            "lights": np.array(self.current_lights, dtype=np.int32)
        }

    def _check_sequence_correct(self):
        
        correct_count = 0
        # Compare robot presses with light sequence element-wise
        for pressed, target in zip(self.robot_press_sequence, self.current_light_sequence):
            if pressed == target:
                correct_count += 1
            else:
                break  # stop counting after first mismatch
        return correct_count


    def step(self, action):
        self.steps += 1

        self.robot_position  = self.robot_model.step(action)
        
        # Reaction to lights (you can define location-specific behavior too)
        reward = 0.0
        terminated = False

        if not self._check_sequence():
            terminated = True
        else:
            
            color_pressed = action - 3
            if color_pressed in self.current_lights:
                reward = 1.0

        terminated = self.steps >= self.max_steps
        truncated = False
        info = {
            "robot_position": self.robot_position,
            "expected_sequence": self.current_light_sequence,
            "steps": self.steps
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Robot at {self.robot_position}, Lights: {self.current_lights}")


    def capture_env(self):
        # Capture the current environment state
        img = self.robot_model.camera_cap()
        if img is not None:
            return img
        else:
            print("Failed to capture image")
            return None

if __name__ == "__main__":
    # test one 
    obs, info = env.reset()
    # test two 
    obs, reward, terminated, truncated, info = env.step(action)
