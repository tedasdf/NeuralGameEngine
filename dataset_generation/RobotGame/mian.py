import os
import sys
import pickle
import json
import logging
import torch
import wandb
from stable_baselines3 import PPO
import zstandard as zstd
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, List
from pydantic import BaseModel
import csv
from huggingface_hub import HfApi

class DataRecord(BaseModel):
    episode: int
    frames: List[Any]
    actions: List[int]
    batch_id: int
    is_last_batch: bool


class DataCollectorCallback(BaseCallback):
    def __init__(self, save_path="./models", enable_rmq=True, save_locally=False, frames_to_skip=4, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.enable_rmq = enable_rmq
        self.save_locally = save_locally
        self.frames_to_skip = frames_to_skip

        os.makedirs(save_path, exist_ok=True)
        self.connection = None
        self.channel = None
        self.redis_client = None
        self.episode_keys_buffer = []
        self.frames_buffer = []
        self.actions_buffer = []
        self.states_buffer = [] 
        self.max_batch_size = 500 * 1024 * 1024  # 500 MB
        self.episode_count = 0
        
        self.api = HfApi(token=os.getenv("HF_TOKEN"))
        
    def _publish_keys_to_queue(self):
        if self.enable_rmq:
            message = json.dumps(self.episode_keys_buffer)
            self.channel.basic_publish(exchange='', routing_key='HF_upload_queue', body=message)
            logging.info(f"Published keys to RabbitMQ queue: {self.episode_keys_buffer}")
            self.episode_keys_buffer.clear()


    # def _save_data_to_redis(self, episode):
    #     """Save current buffer to Redis with compression."""
    #     logging.info("Saving data to Redis")
    #     key = f"episode_{episode}"
    #     data = {'episode': episode, 'frames': self.frames_buffer, 'actions': self.actions_buffer}
    #     serialized_data = pickle.dumps(data)

    #     cctx = zstd.ZstdCompressor()
    #     compressed_data = cctx.compress(serialized_data)

    #     logging.info(f"Original size: {sys.getsizeof(serialized_data)} bytes, Compressed size: {len(compressed_data)} bytes")

    #     self.redis_client.set(key, compressed_data)
    #     self.episode_keys_buffer.append(key)
    #     self.clear_buffers()

    #     if len(self.episode_keys_buffer) >= 20:
    #         self._publish_keys_to_queue()

    def _get_buffer_size(self):
        """Estimate current buffer size in bytes."""
        frame_size = sum(frame.nbytes for frame in self.frames_buffer)
        action_size = sum(sys.getsizeof(action) for action in self.actions_buffer)
        state_size = sum(state.nbytes for state in self.state_buffer)
        return frame_size + action_size + state_size

    def _save_frames_locally(self, episode):
        """Save frames and associated data (action, state) as a table."""
        episode_dir = f"episode_{episode}"
        os.makedirs(episode_dir, exist_ok=True)

        metadata_path = os.path.join(episode_dir, "metadata.csv")
        with open(metadata_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["frame_index", "filename", "action", "state"])  # header

            for idx, (frame, action, state) in enumerate(zip(self.frames_buffer, self.actions_buffer, self.states_buffer)):
                filename = f"{idx:05d}_action_{action}.png"
                filepath = os.path.join(episode_dir, filename)

                # Save image
                Image.fromarray(frame).save(filepath)

                # Write metadata row
                writer.writerow([idx, filename, action, state])

        
        self.api.upload_folder(
            folder_path=episode_dir,  # your local folder path
            repo_id="Sing0402/robot-episodes",  # your dataset repo name
            repo_type="dataset",
        )
        print("Upload done!")

    def clear_buffers(self):
        """Clear the in-memory frame and action buffers."""
        self.frames_buffer.clear()
        self.actions_buffer.clear()
        self.state_buffer.clear()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", None)

        for i, info in enumerate(infos):
            frame = info.get("frame")
            state = info.get('state')
            action = actions[i] if actions is not None else None
            done = info.get("done", False)

            if frame is not None and action is not None:
                self.frames_buffer.append(frame)
                self.actions_buffer.append(action)
                self.states_buffer.append(state)

            # if self.enable_rmq:
            #     buffer_size = self._get_buffer_size()
            #     logging.info(f"Current buffer size: {buffer_size} bytes")
                # if buffer_size >= self.max_batch_size:
                #     logging.warning("Buffer exceeded 500MB, saving to Redis")
                #     self._save_data_to_redis(episode_idx)

            if done:
                logging.info(f"Episode {self.episode_count} done, saving remaining buffer immediately.")
                if self.save_locally:

                    self._save_frames_locally(self.episode_count)
                    self.episode_count += 1
                # if self.enable_rmq:
                #     self._save_data_to_redis(episode_idx)
                self.clear_buffers()

        return True


class TrainLoggingCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info(torch.cuda.memory_summary())

            model_file = os.path.join(self.save_path, f"pacman_{self.num_timesteps}.zip")
            self.model.save(model_file)
            logging.info(f"Model saved at timestep {self.num_timesteps} to {model_file}")

        return True


class WandbCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                wandb.log({
                    "reward": info['episode']['r'],
                    "length": info['episode']['l'],
                    "timesteps": self.num_timesteps
                })
        return True



if __name__ == "__main__":
    from Robot import Robot
    from RobotEnv import RobotEnv
    import numpy as np
    import cv2
    from stable_baselines3 import SAC
    from stable_baselines3.sac.policies import SACPolicy
    from MultiCamFeatureExtractor import CustomFeatureExtractor
    from game import Game
    import gymnasium as gym
    import time
    import os
    import logging

    wandb.init(
        project="robot-sac-training",
        entity="tedlosingyau-monash-university",
        name="run_001",
        config={
            "algorithm": "SAC",
            "total_timesteps": 100_000,
            "features_dim": 256
        },
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    # game_model = Game(
    #     url="ws://localhost:8080",  # Example URL, adjust as necessary 
    #     sequence_length=4  # Example sequence length
    # )

    # robot_model = Robot(
    #     L1 = 6,
    #     L2 = 7, 
    #     L3 = 7, 
    #     serial_port='COM3',
    #     init_position=(0, 0 , 0), # not yet set
    #     test_mode=False
    # )

    # robot_env = RobotEnv(
    #     robot_model,
    #     game_model,
    #     num_colors =4 ,
    #     num_lights =4,
    #     time_limit = 60, # 1 minute
    # )

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # data_collector_cb = DataCollectorCallback( save_path="./models", enable_rmq=False, save_locally=True)
    # train_logger_cb = TrainLoggingCallback(save_freq=5000, save_path="./models")
    # wandb_cb = WandbCallback()

    model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs ,verbose=1)
    model.learn(total_timesteps=100000)

  