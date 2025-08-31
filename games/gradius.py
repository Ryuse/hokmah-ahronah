import os

import cv2
import numpy as np
import retro
import simpleaudio as sa
import torch
from gym import Env
from gym.envs.classic_control import rendering
from gym.spaces import MultiBinary, Box
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

CHECKPOINT_DIR = './train/'
USE_DIR = './use'
LOG_DIR = "../logs/"

torch.cuda.memory_allocated(3)
states = [
    "Level1-1_99_lives"
]

class Gradius(Env):
    def __init__(self, state):

        super().__init__()
        # Specify action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(11)
        # Startup and instance of the game
        self.game = retro.make(game='Gradius-Nes', state=state,
                               record=".",
                               use_restricted_actions=retro.Actions.FILTERED)

        self.viewer = rendering.SimpleImageViewer()

    def reset(self):
        # Return the first frame
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs

        # Create a attribute to hold the score delta
        self.score = 0
        self.enemy_health_taken = 0
        self.health = 0
        self.matches_won = 0
        self.enemy_matches_won = 0
        return obs

    def preprocess(self, observation):
        # Grayscaling
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        # Resize
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        # Add the channels value
        channels = np.reshape(resize, (84, 84, 1))
        return channels

    def step(self, action):
        # Take a step
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)

        frame_delta = obs - self.previous_frame
        self.previous_frame = obs

        return frame_delta, reward, done, info

    def render(self, *args, **kwargs):
        self.game.render()

    def play_sound(self, array, fs):
        sa.play_buffer(array, 2, 2, 44100)

    def close(self):
        self.game.close()
        
    def get_reward(self, info):
        # Reshape the reward function
        reward = info["score"]


        return reward

def try_model():
    state = states[0]
    env = Gradius(state)
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    obs = env.reset()
    model = PPO.load("D:\_Artworks\Art\\2 - vTuber Assets\Hokmah Ahronah Code\games\\retro\gradius\\use/gradius_model_Level1.zip")

    done = False

    while not done:

        action = model.predict(obs, deterministic=False)[0]
        obs, rew, done, info = env.step(action)

        env.render()
        if done:
            obs = env.reset()
    env.close()

def get_newest_file(path):
    list_of_files = os.listdir(path)
    full_path = [f"{path}/{x}" for x in list_of_files]
    model_name = max(full_path, key=os.path.getctime)
    print(f"[LATEST_MODEL] {model_name}")
    return model_name


if __name__ == "__main__":
    try_model()

