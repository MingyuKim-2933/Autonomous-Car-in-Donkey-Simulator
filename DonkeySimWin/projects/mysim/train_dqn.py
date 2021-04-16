import numpy as np
import random
import gym
import gym_donkeycar
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from collections import deque


DONKEY_SIM_PATH = "../DonkeySimMac-race/donkey_sim.app/Contents/MacOS/donkey_sim"
DONKEY_GYM_ENV_NAME = "donkey-generated-roads-v0"
SIM_HOST = "127.0.0.1"
SIM_ARTIFICIAL_LATENCY = 0

THROTTLE = 0.5
STEERING_ACTION_SPACE = [-1.0, -0.5, 0.0, 0.5, 1.0]

TRAIN_MODE = False
MAX_EPISODES = 1000
MAX_STEPS = 1000
BATCH_SIZE = 32
EPOCH = 50
LEARNING_RATE = 0.0001
DISCOUNT_RATIO = 0.99
REPLAY_MEMORY = 5000
SAVE_PATH = './models/dqn.h5'


class DQN:
    def __init__(self, input_size, output_size, l_rate=1e-3, name='main'):
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self._build_network(l_rate=l_rate)

    def _build_network(self, l_rate=1e-3):
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same", input_shape=self.input_size, activation='relu'))
        model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation='relu'))
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.output_size, activation='linear'))
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
        model.compile(optimizer=optimizer, loss='mse')

        self.model = model
        self.optimizer = optimizer

    def copy_to(self, target_dqn):
        target_dqn.model.set_weights(self.model.get_weights())

    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        return self.model.predict(state)

    def update(self, x, y):
        hist = self.model.fit(x, y)
        return hist

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)


def replay_train(main_dqn, target_dqn, train_batch, discount=0.9):
    xs = []
    ys = []
    for state, action, reward, next_state, done in train_batch:
        Q = main_dqn.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Qs1 = target_dqn.predict(next_state)    # this comes from target DQN
            Q[0, action] = reward + discount * np.max(Qs1)

        ys.append(Q)
        xs.append(state)

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    return main_dqn.update(xs, ys)


def bot_play(env, dqn):
    x = env.reset()
    x = x / 255.
    x = rgb2gray(x)

    s_t = np.stack([x, x, x, x], axis=2)

    reward_sum = 0
    done = False
    while not done:
        env.render()
        si = np.argmax(dqn.predict(s_t))
        a = [STEERING_ACTION_SPACE[si], THROTTLE]

        x1, reward, done, _ = env.step(a)
        x1 = x1 / 255.
        x1 = rgb2gray(x1)

        x1 = np.expand_dims(x1, axis=-1)
        s_t = np.append(x1, s_t[:, :, :3], axis=2)

        reward_sum += reward
    print('Total score:', reward_sum)


def rgb2gray(rgb):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    # return np.expand_dims(gray, axis=-1)
    return gray


def e_greedy(e, action_space, q_vector):
    if np.random.rand(1) < e:
        action_idx = random.randrange(len(action_space))
    else:
        action_idx = np.argmax(q_vector)
    return action_idx


# def noisy_action(q_vector, decay_rate):
#     noise_vector = np.random.randn(*q_vector.shape) * decay_rate
#     action = np.argmax(q_vector + noise_vector)
#     return action


def train():
    conf = {
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "car",
        "font_size": 100,
        "exe_path": DONKEY_SIM_PATH,
        "host": SIM_HOST,
        "port": 9091,
        "guid": 0,
        "max_cte": 8,
    }

    env_name = DONKEY_GYM_ENV_NAME
    # delay = SIM_ARTIFICIAL_LATENCY

    env = gym.make(env_name, conf=conf)

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    input_size = list(env.observation_space.shape)
    input_size[-1] = 4  # rgb --> gray
    output_size = len(STEERING_ACTION_SPACE)

    if TRAIN_MODE:

        replay_buffer = deque()
        steps_list = []

        mainDQN = DQN(input_size, output_size, l_rate=LEARNING_RATE, name='main')
        targetDQN = DQN(input_size, output_size, name='target')

        mainDQN.copy_to(targetDQN)

        for episode in range(MAX_EPISODES):
            x = env.reset()
            x = x / 255.
            x = rgb2gray(x)

            s_t = np.stack([x, x, x, x], axis=2)

            done = False
            e = 0.5 / ((episode // 10) + 1)
            step_count = 0

            while not done:
                # select an action
                Qs = mainDQN.predict(s_t)
                steer_idx = e_greedy(e, STEERING_ACTION_SPACE, Qs)
                action = [STEERING_ACTION_SPACE[steer_idx], THROTTLE]

                # get new state and reward
                x1, reward, done, _ = env.step(action)
                x1 = x1 / 255.
                x1 = rgb2gray(x1)

                x1 = np.expand_dims(x1, axis=-1)
                s_t1 = np.append(x1, s_t[:, :, :3], axis=2)

                if done:
                    reward = -10

                # save the experience to our buffer
                replay_buffer.append((s_t, steer_idx, reward, s_t1, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                # update state
                s_t = s_t1

                step_count += 1
                if step_count > MAX_STEPS:
                    break

            print('Episode: {}, steps: {}'.format(episode, step_count))
            steps_list.append(step_count)

            if len(steps_list) > 30 and np.mean(steps_list[-30:]) > MAX_STEPS:
                break

            # train DQN (50 steps for every 10 episodes)
            if episode > 0 and episode % 10 == 0:
                print('Training DQN ... ', end='', flush=True)
                env.reset()
                for i in range(EPOCH):
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    replay_train(mainDQN, targetDQN, minibatch, DISCOUNT_RATIO)

                # copy network
                mainDQN.copy_to(targetDQN)

        # save model
        mainDQN.save(SAVE_PATH)

        # training history
        print('Maximum reward sum:', max(steps_list))
        print('Average reward sum:', sum(steps_list) / len(steps_list))

    else:
        mainDQN = DQN(input_size, output_size, l_rate=LEARNING_RATE, name='main')
        mainDQN.load(SAVE_PATH)

    # test
    bot_play(env, mainDQN)
    env.close()


if __name__ == '__main__':
    train()

