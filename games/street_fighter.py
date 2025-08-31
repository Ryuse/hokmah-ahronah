import time

import cv2

import numpy as np
import pyaudio
import retro
import scripts.vtube_studio as vtube

import simpleaudio as sa

from gym import Env
from gym.envs.classic_control import rendering
from gym.spaces import MultiBinary, Box

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

CHECKPOINT_DIR = 'retro/street_fighter/train/'
LOG_DIR = "retro/street_fighter/logs/"

fight_states = [
    "rg",
    "rk",
    "rc",
    #"rcar",
    "rz",
    "rd",
    "rr",
    #"rbricks",
    "rh",
    "rb",
    "rbal",
    #"rbarrels",
    "rv",
    "rs",
    "rm"
]

def get_time():
    return time.time()

# Create custom environment
frame_time = 1.0 / 90.0

class StreetFighter(Env):
    def __init__(self, state):
        super().__init__()
        # Specify action space and observation space
        # self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        # Startup and instance of the game
        self.game = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis",
                               state=state,
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
        self.total_rewards = 0

        self.info = {}
        self.last_time = get_time()
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=2,
                        rate=44100,
                        output=True)
        return obs

    def preprocess(self, observation):
        # Grayscaling
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84, 84, 1))

        return channels

    def step(self, action):
        # Take a step
        # # print(action)

        # action = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]) #Bottom Block
        # action = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # Jump
        # action = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) #Duck
        # action = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) #Left
        # action = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]) #Right
        # action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) #C

        obs, reward, done, info = self.game.step(action)

        obs = self.preprocess(obs)

        # make sure every frame is of length 1/60s
        current_time = get_time()
        time_diff = current_time - self.last_time
        if time_diff < frame_time:
            time.sleep(frame_time - time_diff)
        self.last_time = current_time

        a = self.game.em.get_audio()

        bytes_array = a.flatten().astype('int16')
        frames_count = len(a)
        self.stream.write(frames=bytes_array, num_frames=frames_count)

        # Frame delta
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs

        damage_taken = self.health - info["health"]
        if damage_taken > 0:
            vtube.functions["damage"]()
        self.health = info["health"]
        #
        health_taken = self.enemy_health_taken - info["enemy_health"]
        if health_taken > 0:
            vtube.functions["nod_down"]()
        self.enemy_health_taken = info["enemy_health"]

        reward += self.get_reward(info)
        self.render()


        return frame_delta, reward, done, info

    def render(self, *args, **kwargs):

        self.game.render()

    def close(self):
        self.game.close()
        self.stream.close()
        self.p.terminate()

    def play_sound(self, array, fs):

        sa.play_buffer(array, 2, 2, 44100)

    def get_reward(self, info):
        # Reshape the reward function
        reward = 0

        if self.score == 0:
            self.score = info['score']

        reward = int(((info['score'] - self.score) * 0.01))
        self.score = info['score']

        # Encourage fighting
        health_taken = self.enemy_health_taken - info["enemy_health"]
        if health_taken > 0:
            reward += 1
        self.enemy_health_taken = info["enemy_health"]

        # damage_taken = self.health - info["health"]
        # if damage_taken > 0:
        #     reward -= damage_taken * 2
        # self.health = info["health"]

        # Encourages winning
        matches_won = self.matches_won - info["matches_won"]
        if matches_won > 0:
            reward += matches_won * 200
        self.matches_won = info["matches_won"]

        # # Discourages losing
        # enemy_matches_won = self.enemy_matches_won - info["enemy_matches_won"]
        # if enemy_matches_won > 0:
        #     reward -= enemy_matches_won * 100
        # self.enemy_matches_won = info["enemy_matches_won"]

        # # #Encourage doing things faster
        # reward -= 50

        # duck_action = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1])  # Duck
        # jump_action_1 = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1])  # Jump
        # jump_action_2 = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0])  # Jump
        #
        # no_input_1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # No Input
        # no_input_2 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # No Input
        # jump_right = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1]) # JUMP RIGHT
        # move_left = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0]) #MOVE LEFT
        # move_right = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1]) # MOVE RIGHT
        #
        # if np.array_equal(jump_action_1, action) or np.array_equal(jump_action_2, action)\
        #         or np.array_equal(no_input_1, action) or np.array_equal(no_input_2, action):
        #     reward -= 100
        # elif np.array_equal(jump_right, action) or np.array_equal(move_left, action)\
        #         or np.array_equal(move_right, action):
        #     reward += 100
        return reward

def main(state_list):
    print(state_list)
    state = state_list[0]
    print("Starting new model instance")
    env = StreetFighter(state)
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    done = False
    obs = env.reset()

    if state == "rs":
        model = PPO.load("D:\_Artworks\Art\\2 - vTuber Assets\Hokmah Ahronah Code\games\\retro\street_fighter\\backup\SF2_model_rs 2nd best.zip")
    else:
        model = PPO.load(f"./games/retro/street_fighter/use/SF2_model_{state.replace('.', '_')}")

    info_matches_won = 0
    total_matches_won = 0
    total_reward = 0
    while not done:
        try:
            action = model.predict(obs, deterministic=False)[0]

            obs, reward, done, info = env.step(action)

            total_reward += reward
            if info[0]["matches_won"] != info_matches_won:
                if info[0]["matches_won"] >= 2:
                    total_matches_won += 1
                    print(f"Won {total_matches_won} matches at state: {state}")
                info_matches_won = info[0]["matches_won"]

            env.render()
            if done:
                print(f"[REWARD] {total_reward}")
                print(info[0])


        except Exception as e:
            print(f"[main EXCEPTION] {e}")

    env.close()
    if total_matches_won > 0:
        for i in range(total_matches_won):
            state_list.pop(0)

    if state_list == []:
        state_list = fight_states.copy()

    return state_list


def look_at_all_state():
    current_state = fight_states.copy()
    while 1:
        current_state = main(current_state)

if __name__ == "__main__":
    while 1:
        try:
            look_at_all_state()
            main("rg")
        except Exception as e:
            print(f"[init EXCEPTION] {e}")
