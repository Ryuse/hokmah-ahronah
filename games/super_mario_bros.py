import retro
import numpy as np
import time
import cv2
import os
import torch

from gym import Env
from gym.envs.classic_control import rendering
from gym.spaces import MultiBinary, Box

from matplotlib import pyplot as plt

from stable_baselines3 import PPO

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecEnv, VecEnvWrapper

from sb3_contrib import RecurrentPPO

CHECKPOINT_DIR = './train/'
USE_DIR = './use'
LOG_DIR = "../logs/"

torch.cuda.memory_allocated(3)
states = [
    "Level1-1",
    "Level2-1",
    "Level3-1",
    "Level4-1",
    "Level5-1",
    "Level6-1",
    "Level7-1",
    "Level8-1",


]


class SuperMarioBros(Env):
    def __init__(self, state):

        super().__init__()
        # Specify action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(11)

        # Startup and instance of the game
        self.game = retro.make(game='SuperMarioBros-Nes', state=state,
                               use_restricted_actions=retro.Actions.FILTERED)

        self.viewer = rendering.SimpleImageViewer()

    def reset(self):
        # Return the first frame
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs

        # info = {'levelLo': 0, 'xscrollHi': 0, 'levelHi': 0, 'coins': 0, 'xscrollLo': 0, 'time': 0, 'scrolling': 16, 'lives': -1, 'score': 0}

        # Create a attribute to hold the score delta
        self.score = 0
        self.xscrollHi = 0
        self.xscrollLo = 0

        self.info = {}
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
        # print(type(action))
        # print(action)
        #action = np.array([0, 0, 0, 0, 1, 0, 0, 0])  # Jump
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)

        # if info != self.info:
        #     print(info)
        #     self.info = info

        #action = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Right
        # Frame delta
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs

        reward += self.get_reward(info)

        return frame_delta, reward, done, info

    def render(self, *args, **kwargs):
        self.game.render()

    def close(self):
        self.game.close()

    def get_reward(self, info):
        # Reshape the reward function
        reward = 0

        if self.xscrollLo != info["xscrollLo"]:
            reward += info["xscrollLo"] - self.xscrollLo
            self.xscrollLo = info["xscrollLo"]



        return reward


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, learning_rate, log_dir, state, verbose=1, RPPO=False):
        super(TrainAndLoggingCallback, self).__init__(verbose)

        self.check_freq = check_freq
        self.save_path = save_path
        self.log_dir = log_dir

        self.learning_rate = learning_rate
        self.state = state
        self.RPPO = RPPO

        self.best_mean_reward = -np.inf

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")

            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-10:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model

                    use_name = f"mario_model_{self.state}"
                    use_path = os.path.join(USE_DIR, use_name)

                    if self.verbose >= 1:
                        print(f"Saving new best model to {use_path}")

                    self.model.save(use_path)

                    # best_model_name =
                    # self.model.save(self.save_path)

            list_of_files = os.listdir(self.save_path)
            full_path = [f"{self.save_path}/{self.state}/{x}" for x in list_of_files]

            if len(list_of_files) >= 10:
                oldest_file = min(full_path, key=os.path.getctime)
                os.remove(oldest_file)

            if self.RPPO:
                model_name = f"mario_model_RPPO_{self.n_calls}"
            else:
                model_name = f"mario_model_{self.n_calls}"
            train_path = os.path.join(f"{self.save_path}/{self.state}/", model_name)
            self.model.save(train_path)

        return True

def train(state, learning_rate):
    # Create environment
    env = SuperMarioBros(state)
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    callback = TrainAndLoggingCallback(check_freq=8000,
                                       save_path=CHECKPOINT_DIR,
                                       learning_rate=learning_rate,
                                       log_dir=LOG_DIR,
                                       state=state)

    try:
        model_name = f"{USE_DIR}/mario_model_{state}"
        custom_objects = {
            'learning_rate': learning_rate,
            'n_steps': 5120,
            'batch_size': 256
        }
        model = PPO.load(model_name, custom_objects=custom_objects)
        model.set_env(env)
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1,
                    learning_rate=learning_rate, n_steps=5120, batch_size=256,
                    # gamma=0.9085173,
                    # clip_range=0.3910507,
                    # gae_lambda=0.83766
                    )

    model.learn(total_timesteps=640000, callback=callback)
    env.close()


# lose list = [rk, rz, rr, ]

# info = {'levelLo': 0, 'xscrollHi': 0, 'levelHi': 0, 'coins': 0, 'xscrollLo': 0, 'time': 0, 'scrolling': 16, 'lives': -1, 'score': 0}


def try_model():
    state = states[0]
    env = SuperMarioBros(state)
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    obs = env.reset()

    model = PPO.load("D:\_Artworks\Art\\2 - vTuber Assets\Hokmah Ahronah Code\games\\retro\super_mario_bros\\use/mario_model_Level1-1")
    #model = PPO.load("D:\_Artworks\Art\\2 - vTuber Assets\Hokmah Ahronah Code\games\\retro\super_mario_bros\\use/backups/mario_model_Level1-1")
    prev_info = {}
    while True:
        # action_space will by MultiBinary(16) now instead of MultiBinary(8)
        # the bottom half of the actions will be for player 1 and the top half for player 2

        # action = env.action_space.sample()
        # print(action)
        action = model.predict(obs, deterministic=False)[0]
        #print(action)
        obs, rew, done, info = env.step(action)
        #
        # if info != prev_info:
        #     print(info)

        # rew will be a list of [player_1_rew, player_2_rew]
        # done and info will remain the same
        env.render()
        if done:

            obs = env.reset()
    env.close()

if __name__ == "__main__":


    try_model()

