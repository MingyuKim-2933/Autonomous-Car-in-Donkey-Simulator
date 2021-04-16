import os
import random
import argparse
import uuid

import numpy as np
import gym
import cv2
import cloudpickle

from collections import deque
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Activation, Flatten, MaxPooling2D, Dropout
import tensorflow as tf

import gym_donkeycar


DONKEY_SIM_PATH = "../DonkeySimMac-race/donkey_sim.app/Contents/MacOS/donkey_sim"
DONKEY_GYM_ENV_NAME = "donkey-generated-roads-v0"
SIM_HOST = "127.0.0.1"
SIM_ARTIFICIAL_LATENCY = 0

VAE_WEIGHT_PATH = './vae/vae-level-0-dim-32.pkl'
MODEL_PATH = 'models/ddqn_vae.h5'

EPISODES = 3000
img_rows, img_cols = 80, 160
img_channels = 3    # rgb
z_size = 32


# for printing numpy float
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})


class DoubleDQN_VAE_Agent:

    def __init__(self, state_size, action_space, image_size, vae_weights=None, train=True):
        self.t = 0
        self.Q = None
        self.train = train

        # Get size of state and action
        self.state_size = state_size
        self.action_space = action_space
        self.image_size = image_size
        self.vae_weights = vae_weights

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if self.train:
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = 100
        self.explore = 10000

        # Create replay memory using deque
        self.memory = deque(maxlen=10000)

        # Create VAE encoder model
        self.encoder = self.build_vae_encoder()

        # Create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

    def build_vae_encoder(self):
        encoder = Sequential()
        encoder.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu', name='enc_conv1',
                           input_shape=self.image_size))
        encoder.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', name='enc_conv2'))
        encoder.add(Conv2D(128, (4, 4), strides=(2, 2), activation='relu', name='enc_conv3'))
        encoder.add(Conv2D(256, (4, 4), strides=(2, 2), activation='relu', name='enc_conv4'))
        encoder.add(Flatten())  # shape: [-1, 3 * 8 * 256]
        encoder.add(Dense(z_size, name='enc_fc_mu'))

        if self.vae_weights:
            encoder.get_layer('enc_conv1').set_weights(self.vae_weights[0:2])
            encoder.get_layer('enc_conv2').set_weights(self.vae_weights[2:4])
            encoder.get_layer('enc_conv3').set_weights(self.vae_weights[4:6])
            encoder.get_layer('enc_conv4').set_weights(self.vae_weights[6:8])
            encoder.get_layer('enc_fc_mu').set_weights(self.vae_weights[8:10])
            encoder.trainable = False

        return encoder

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=self.state_size))
        model.add(Dense(15, activation="linear"))   # 15 categorical bins for Steering angles

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    def encode_image(self, obs):
        obs = cv2.resize(obs, (img_cols, img_rows))
        obs = obs / 255.
        obs -= 0.5
        obs *= 2.
        obs = np.expand_dims(obs, axis=0)
        return self.encoder.predict(obs)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, s_t):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()[0]
        else:
            # print("Return Max Q Prediction")
            q_value = self.model.predict(s_t)
            self.Q = q_value[0]

            # Convert q array to steering value
            return linear_unbin(q_value[0])

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)

        targets = self.model.predict(state_t)
        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])

        self.model.train_on_batch(state_t, targets)

    def load_model(self, name):
        self.model.load_weights(name)

    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


## Utils Functions ##

def linear_bin(a):
    """
    Convert a value to a categorical array.

    Parameters
    ----------
    a : int or float
        A value between -1 and 1

    Returns
    -------
    list of int
        A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    """
    Convert a categorical array to value.

    See Also
    --------
    linear_bin
    """
    if not len(arr) == 15:
        raise ValueError('Illegal array length, must be 15')
    b = np.argmax(arr)
    a = b * (2 / 14) - 1
    return a


def run_ddqn(args):
    '''
    run a DDQN training session, or test it's result, with the donkey simulator
    '''

    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,

        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "ddqn+vae",
        "font_size": 80,

        "racer_name": "DDQN",
        "country": "KOR",
        "bio": "Learning to drive w DDQN RL",
        "guid": str(uuid.uuid4()),

        "max_cte": 3,
    }

    # Construct gym environment. Starts the simulator if path is given.
    env = gym.make(args.env_name, conf=conf)

    # Get size of state and action from environment
    image_size = (img_rows, img_cols, img_channels)
    state_size = (z_size, )
    action_space = env.action_space  # Steering and Throttle

    try:

        with open(args.vae, 'rb') as file:
            data, params = cloudpickle.load(file)

        agent = DoubleDQN_VAE_Agent(state_size, action_space, image_size, vae_weights=params, train=not args.test)

        throttle = args.throttle  # Set throttle as constant value

        if os.path.exists(args.model):
            print("load the saved model")
            agent.load_model(args.model)

        for e in range(EPISODES):
            print("Episode: ", e)

            episode_len = 0

            done = False
            obs = env.reset()

            s_t = agent.encode_image(obs)   # (1, 32)

            while not done:

                # Get action for the current state and go one step in environment
                steering = agent.get_action(s_t)
                action = [steering, throttle]
                next_obs, reward, done, info = env.step(action)

                s_t1 = agent.encode_image(next_obs)

                if agent.train:
                    # Save the sample <s, a, r, s'> to the replay memory
                    agent.replay_memory(s_t, np.argmax(linear_bin(steering)), reward, s_t1, done)
                    agent.update_epsilon()

                    agent.train_replay()

                s_t = s_t1
                agent.t = agent.t + 1
                episode_len = episode_len + 1
                if agent.t % 30 == 0:
                    print("EPISODE", e, "TIMESTEP", agent.t, "/ ACTION", action, "/ REWARD", reward, "/ EPISODE LENGTH",
                          episode_len, "/\n    Q ", agent.Q)

                if done:
                    if agent.train:
                        # Every episode update the target model to be same with model
                        agent.update_target_model()

                        # Save model for each episode
                        agent.save_model(args.model)

                    print("episode:", e, "  memory length:", len(agent.memory),
                          "  epsilon:", agent.epsilon, " episode length:", episode_len)

    except KeyboardInterrupt:
        print("stopping run...")
    finally:
        env.unwrapped.close()


if __name__ == "__main__":
    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0"
    ]

    parser = argparse.ArgumentParser(description='ddqn')
    parser.add_argument('--sim', type=str, default=DONKEY_SIM_PATH,
                        help='path to unity simulator')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='path to model')
    parser.add_argument('--vae', type=str, default=VAE_WEIGHT_PATH, help='path to vae encoder weights (.pkl)')
    parser.add_argument('--test', action="store_true", help='agent uses learned model to navigate env')
    parser.add_argument('--port', type=int, default=9091, help='port to use for websockets')
    parser.add_argument('--throttle', type=float, default=0.5, help='constant throttle for driving')
    parser.add_argument('--env_name', type=str, default=DONKEY_GYM_ENV_NAME, help='name of donkey sim environment',
                        choices=env_list)

    args = parser.parse_args()

    run_ddqn(args)

