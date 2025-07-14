import gymnasium as gym
from gymnasium import spaces
import serial
import numpy as np
import time
import pyzed.sl as sl

class RobotEnv(gym.Env):
    
    def __init__(
            self,
            robot_model,
            game_model,
            num_colors,
            num_lights,
            serial_port,
            baud_rate = 9600,
            timeout = 1,
        ):
        super(RobotEnv, self).__init__()
        
        # robot and game
        self.robot_model = robot_model
        self.game_model = game_model
        # Define action and observation space
        

        # Observation space ( 0 : no light , 1 : red , 2 : green , 3 : blue)
        self.num_colors = num_colors 
        self.num_lights = num_lights

        
        self.observation_space = spaces.Dict({
            "camera": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            "robot_state": spaces.Box(low=-5, high=15, shape=(3,), dtype=np.float32)
        })

        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(3,), dtype=np.float32)

        
        self.max_steps = 1000  # Maximum number of steps per episode
        self.reward = None
        self.steps = None

        # Camera
        self.zed = sl.Camera()
        devices = self.zed.get_device_list()
        if not devices:
            raise ValueError("No ZED camera found.")
        self.zed_serial = devices[:].serial_number
        self._init_camera()

    def _init_camera(self):
        # Initialize the ZED camera
        self.zed1 = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30

        input_type = sl.InputType()
        input_type.set_from_serial_number(self.zed_serial)

        init_params.input = input_type

        status = self.zed1.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Error opening camera: {status}")
            exit(1)
        print("Camera opened successfully.")
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_model.reset()
        self.current_sequence = [np.random.randint(1, self.num_colors) for _ in range(self.num_lights)]
        self.steps = 0
        self.reward = 0.0
        # send the current sequence to hardware
        if  self.game_model.connected & self.game_model.active == False:
            self.game_model.reset(self.current_sequence)
        
        info = {
            "robot_position": self.robot_model.state[0:3,3],
            "expected_sequence": self.game_model.sequence,
            "player_sequence": self.game_model.player_sequence,
            "steps": self.steps,
            "reward": self.reward,

        }

        return self._get_obs() , info

    def _get_obs(self):
        if self.zed1.grab() == sl.ERROR_CODE.SUCCESS:
            left_image = sl.Mat()
            self.zed1.retrieve_image(left_image, sl.VIEW.LEFT)

            left_image_np = left_image.get_data()[:, :, :3]  # Drop alpha if needed
           
        return {
            "camera": left_image_np,
            "pos": np.array(self.robot_model.state[0:3, 3], dtype=np.int32),
        }
    
    def _compute_reward(self, robot_pos):

        reward , terminated = self.game_model.check_sequence(robot_pos)

        return reward, terminated
    
    def step(self, action):
        self.steps += 1
    
        valid  = self.robot_model.step(action)        
        
        # reward function 
        reward, terminated = self._compute_reward(self.robot_model.state[0:3, 3])

        terminated = terminated or (self.steps >= self.max_steps) or (not valid)
        truncated = (self.steps >= self.max_steps)

        observation = self._get_obs()

        info = {
            "robot_position": self.robot_model.state[0:3,3],
            "expected_sequence": self.game_model.sequence,
            "player_sequence": self.game_model.player_sequence,
            "steps": self.steps,
            "camera": observation['camera'],
            "position": observation['pos'],
            "action": action,
            "done" : terminated or truncated
        }

        return self._get_obs(), reward, terminated, truncated, info # apply action and return (obs, reward, done, info)
    
    def render(self, mode="human"):
        print(f"Robot at {self.robot_position}, Lights: {self.current_lights}")
    


if __name__ == "__main__":
    from game import Game
    from Robot import Robot
    
    # Your ESP32 WebSocket port
    url = f"ws://192.168.0.100:81"

    game = Game(url, sequence_length=4)

    robot_model = Robot(L1=7.14, L2=7, L3=8.34 , x_offset=0.9185 , y_offset=0, z_offset=0, serial_port='COM3', test_mode=False)
    
    # # Example usage - replace ... with actual robot_model and game_model instances
    env = RobotEnv(robot_model= robot_model,game_model= game,num_colors=4,num_lights=4)

    check_env(env)

    # obs, info = env.reset()
    # # action = env.action_space.sample()
    # # print("Initial Action:", action)
    # # obs, reward, terminated, truncated, info = env.step(action)

    # # print("Observation keys:", obs.keys())
    # # print("Reward:", reward)
    # # print("Terminated:", terminated)
    # # print("Truncated:", truncated)
    # # print("Info:", info)


    # for i in range(10):
    #     obs ,  =  env.step()

