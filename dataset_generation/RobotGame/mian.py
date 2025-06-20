

if __name__ == "__main__":
    from Robot import Robot
    from RobotEnv import RobotEnv
    import numpy as np
    import cv2
    from stable_baselines3 import SAC

    import gymnasium as gym

    env = RobotEnv(
        robot_model=Robot(),
        num_colors=3,  # Example: Red, Green, Blue
        num_lights=5,  # Example: 5 lights
        time_limit=10,  # Example: 10 seconds time limit
        serial_port="/dev/ttyUSB0",  # Adjust as necessary
        baud_rate=9600,
        timeout=1,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1000)
    

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4) # change this into a loop


    model.save("RobotGame_SAC_Model")



## sample code to do the pipeline of uploading dataset 
# def train(self):
#     if self.enable_rmq:
#         self._setup_rabbitmq()

#     screen = self.env.reset(mode='rgb_array')
#     n_actions = self.env.action_space.n

#     self.agent = PacmanAgent(screen.shape, n_actions)
#     self.memory = ProportionalPrioritizedReplayBuffer(100000)  # Use the new prioritized replay buffer

#     frames_buffer, actions_buffer = [], []
#     max_batch_size = 500 * 1024 * 1024  # 400 MB

#     for i_episode in range(self.episodes):
#         state = self.env.reset(mode='rgb_array')
#         ep_reward = 0.
#         epsilon = self._get_epsilon(i_episode)
#         logging.info("-----------------------------------------------------")
#         logging.info(f"Starting episode {i_episode} with epsilon {epsilon}")

#         for t in count():
#             current_frame = self.env.render(mode='rgb_array')
#             self.env.render(mode='human')

#             action = self.agent.select_action(state, epsilon, n_actions)
#             next_state, reward, done, _ = self.env.step(action)
#             reward = max(-1.0, min(reward, 1.0))
#             ep_reward += reward
            
#             if self.enable_rmq or self.save_locally:
#                 frames_buffer.append(current_frame)
#                 actions_buffer.append(self.action_encoder(action))

#             self.memory.cache(state, next_state, action, reward, done)

#             state = next_state if not done else None

#             self.agent.optimize_model(self.memory, n_steps=3)
#             if done:
#                 pellets_left = self.env.maze.get_number_of_pellets()
#                 if self.save_locally:
#                     self._save_frames_locally(frames=frames_buffer, episode=i_episode, actions=actions_buffer)
#                 logging.info(f"Episode #{i_episode} finished after {t + 1} timesteps with total reward: {ep_reward} and {pellets_left} pellets left.")
                
#                 # Log the reward to wandb
#                 wandb.log({"episode": i_episode, "reward": ep_reward, "pellets_left": pellets_left})
                
#                 break

#             # Check if the batch size limit is reached
#         if self.enable_rmq:
#             buffer_size = self._get_buffer_size(frames_buffer, actions_buffer)
#             logging.info(f"Buffer size: {buffer_size} bytes")
#             if buffer_size >= max_batch_size:
#                 logging.warning("BUFFER SIZE EXCEEDING 500MB")
#             self._save_data_to_redis(i_episode, frames_buffer, actions_buffer)
#             frames_buffer, actions_buffer = [], []
#             # batch_id += 1

#         # Send remaining data at the end of the episode
#         if frames_buffer and self.enable_rmq:
#             self._save_data_to_redis(i_episode, frames_buffer, actions_buffer)
#             frames_buffer, actions_buffer = [], []

#         if i_episode > 2: 
#             if i_episode % 10 == 0:
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
#                     logging.info(torch.cuda.memory_summary())
#                 torch.autograd.set_detect_anomaly(True)

#             if i_episode % 1000 == 0:
#                 self.agent.save_model('pacman.pth')
#                 logging.info(f"Saved model at episode {i_episode}")

#     logging.info('Training Complete')
#     self.env.close()
#     self.agent.save_model('pacman.pth')
#     if self.enable_rmq:
#         self._close_rabbitmq()

# def _get_buffer_size(self, frames_buffer, actions_buffer):
#     # Estimate the size of the buffers in bytes
#     buffer_size = sum([frame.nbytes for frame in frames_buffer]) + \
#                     sum([sys.getsizeof(action) for action in actions_buffer])
#     return buffer_size

# def _get_epsilon(self, frame_idx):
#     # Start with a lower initial epsilon and decay faster
#     initial_epsilon = 0.8  # Lower initial exploration rate
#     min_epsilon = 0.05      # Minimum exploration rate
#     decay_rate = 5e2       # Faster decay rate

#     return min_epsilon + (initial_epsilon - min_epsilon) * math.exp(-1. * frame_idx / decay_rate)

# def _save_data(self, data_record: DataRecord):
#     self.save_queue.put(data_record)
#     if self.enable_rmq:
#         self._publish_to_rabbitmq(self.save_queue.get())

# def _save_remaining_data(self, data_record: DataRecord):
#     if data_record.frames:
#         self._save_data(data_record)

# def _publish_to_rabbitmq(self, data: DataRecord):
#     import pickle

#     # Serialize the data using pickle
#     message = pickle.dumps(data.dict())

#     # Publish the message to the queue
#     self.channel.basic_publish(exchange='',
#                                 routing_key='HF_upload_queue',
#                                 body=message)

#     logging.info("Published dataset to RabbitMQ queue 'HF_upload_queue'")

# def _save_frames_locally(self, frames, episode, actions):
#     # Create a directory for the episode if it doesn't exist
#     episode_dir = f"episode_{episode}_frs{self.frames_to_skip}"
#     if not os.path.exists(episode_dir):
#         os.makedirs(episode_dir)

#     # Save each frame as a PNG file with the episode and action in the filename
#     for idx, frame in enumerate(frames):
#         action = actions[idx]
#         filename = os.path.join(episode_dir, f"{idx:05d}.png")
#         Image.fromarray(frame).save(filename)
#         # logging.info(f"Saved frame {idx} of episode {episode} with action {action} to {filename}")
