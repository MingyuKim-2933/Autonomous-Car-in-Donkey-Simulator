import os
import argparse
import uuid

import numpy as np
import gym
import gym_donkeycar
import cloudpickle
import tensorflow as tf
import matplotlib.pyplot as plt

from soft_actor_critic import SAC_VAE_Agent, SAC_Agent


DONKEY_SIM_PATH = "/Users/brix/Works/SideProjects/UnityRL/DonkeySimMac-race/donkey_sim.app/Contents/MacOS/donkey_sim"
SIM_HOST = "127.0.0.1"
SIM_ARTIFICIAL_LATENCY = 0

DONKEY_GYM_ENV_NAME = "donkey-generated-roads-v0"
VAE_WEIGHT_PATH = './vae/vae-level-0-dim-32.pkl'
PW_PATH = 'models/sac_gen-roads_vae_p.h5'
Q1W_PATH = 'models/sac_gen-roads_vae_q1.h5'
Q2W_PATH = 'models/sac_gen-roads_vae_q1.h5'

# DONKEY_GYM_ENV_NAME = "donkey-minimonaco-track-v0"
# VAE_WEIGHT_PATH = None
# PW_PATH = 'models/sac_monaco_ce_p.h5'
# Q1W_PATH = 'models/sac_monaco_ce_q1.h5'
# Q2W_PATH = 'models/sac_monaco_ce_q2.h5'


EPISODES = 1000
EPOCH = 30
img_rows, img_cols = 80, 160
img_channels = 3    # rgb
z_size = 32

# img_rows, img_cols = 40, 40
# img_channels = 1    # gray


# for printing numpy float
np.set_printoptions(formatter={'float_kind': "{:.2f}".format})

tf.keras.backend.set_floatx('float64')


def run_sac(args):
    '''
    run a DDQN training session, or test it's result, with the donkey simulator
    '''

    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "sac+vae",
        "font_size": 80,
        "racer_name": "SAC",
        "country": "KOR",
        "bio": "Learning to drive w SAC RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 5.0
    }

    # Construct gym environment. Starts the simulator if path is given.
    env = gym.make(args.env_name, conf=conf)

    # Get size of state and action from environment
    image_size = (img_rows, img_cols, img_channels)
    action_size = 1

    try:
        if args.vae:
            with open(args.vae, 'rb') as file:
                data, params = cloudpickle.load(file)

        agent = SAC_VAE_Agent(image_size, action_size, train=not args.test, z_size=z_size, vae_weights=params)
        # agent = SAC_Agent(image_size, action_size, train=not args.test)
        agent.summary()

        throttle = args.throttle  # Set throttle as constant value

        if not agent.train and os.path.exists(args.pw) and os.path.exists(args.q1w) and os.path.exists(args.q2w):
            print("load the saved models")
            agent.load_models(args.pw, args.q1w, args.q2w)

        for e in range(EPISODES):
            print("Episode: ", e)

            episode_len = 0

            done = False
            obs = env.reset()
            s_t = agent.preprocess_image(obs)   # (1, 32)
            losses = None
            # plt.imshow(s_t.squeeze())
            # plt.show()

            while not done:
                if episode_len < 15:
                    action = [0, throttle]
                    next_obs, reward, done, info = env.step(action)
                    s_t1 = agent.preprocess_image(next_obs)

                else:
                    # Get action for the current state and go one step in environment
                    steering, _ = agent.get_action(s_t)
                    steering = steering.numpy()[0, 0]

                    action = [steering, throttle]
                    next_obs, reward, done, info = env.step(action)
                    s_t1 = agent.preprocess_image(next_obs)

                    if agent.train:
                        if done:
                            reward = -10.

                        # Save the sample <s, a, r, s'> to the replay memory
                        agent.replay_memory(
                            s_t,
                            np.array(steering, dtype=np.float64).reshape((1, 1)),
                            np.array(reward, dtype=np.float64).reshape((1, 1)),
                            s_t1,
                            np.array(done, dtype=np.float64).reshape((1, 1)))

                        # train one batch (including target update)
                        losses = agent.train_replay()

                    if episode_len % 30 == 0 and episode_len > 0:
                        print("EPISODE", e, "/ EPISODE LENGTH", episode_len, "/ ACTION", action, "/ REWARD", reward,
                              "/ TIMESTEP", agent.t)
                        # plt.imshow(s_t.squeeze())
                        # plt.show()
                        if agent.train and losses:
                            print('loss:', *losses)

                    if done:
                        print(">> EPISODE", e, "/ EPISODE LENGTH", episode_len, "/ memory length", len(agent.memory))
                        if agent.train:
                            # Save model for each episode
                            agent.save_models(args.pw, args.q1w, args.q2w)

                s_t = s_t1
                agent.t = agent.t + 1
                episode_len = episode_len + 1

            # train
            if agent.train and agent.ready_for_training():
                # _ = env.reset()
                # for i in range(EPOCH):
                #     losses = agent.train_replay()
                #     print('epoch:', i, ' loss:', *losses)

                # Save model
                agent.save_models(args.pw, args.q1w, args.q2w)

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

    parser = argparse.ArgumentParser(description='sac')
    parser.add_argument('--sim', type=str, default=DONKEY_SIM_PATH,
                        help='path to unity simulator')
    parser.add_argument('--pw', type=str, default=PW_PATH, help='path to P model')
    parser.add_argument('--q1w', type=str, default=Q1W_PATH, help='path to Q1 model')
    parser.add_argument('--q2w', type=str, default=Q2W_PATH, help='path to Q2 model')
    parser.add_argument('--vae', type=str, default=VAE_WEIGHT_PATH, help='path to vae encoder weights (.pkl)')
    parser.add_argument('--test', action="store_true", help='agent uses learned model to navigate env')
    parser.add_argument('--port', type=int, default=9091, help='port to use for websockets')
    parser.add_argument('--throttle', type=float, default=0.5, help='constant throttle for driving')
    parser.add_argument('--env_name', type=str, default=DONKEY_GYM_ENV_NAME, help='name of donkey sim environment',
                        choices=env_list)

    args = parser.parse_args()

    run_sac(args)

