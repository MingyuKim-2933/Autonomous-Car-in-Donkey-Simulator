from collections import deque
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Concatenate, BatchNormalization
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
import cv2


class SAC_Agent:

    def __init__(self, state_size, action_size, train=True):
        self.t = 0
        self.train = train

        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # Crop image
        self.cropped_rows = int(self.state_size[0] * 0.6)

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.lr_q = 1e-4
        self.lr_p = 1e-4
        self.lr_a = 1e-3
        self.batch_size = 32
        self.train_start = 300
        self.epsilon = 1e-9
        self.target_entropy = -1.0
        self.tau = 0.01

        # Create replay memory using deque
        self.memory = deque(maxlen=20000)

        # Create main model and target model
        self.pi = self.build_Pnet()
        self.q1 = self.build_Qnet()
        self.q2 = self.build_Qnet()
        self.q1_target = self.build_Qnet()
        self.q2_target = self.build_Qnet()

        self.alpha = tf.Variable(0.01, dtype=tf.float64)

        self.actor_optimizer = tf.keras.optimizers.Adam(self.lr_p)
        self.critic1_optimizer = tf.keras.optimizers.Adam(self.lr_q)
        self.critic2_optimizer = tf.keras.optimizers.Adam(self.lr_q)
        self.alpha_optimizer = tf.keras.optimizers.Adam(self.lr_a)

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.hard_update_target_models()

    def preprocess_image(self, obs):
        img_rows, img_cols = self.state_size[0:2]
        obs = cv2.resize(obs, (img_cols, img_rows))
        # obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # obs = cv2.Canny(obs, 180, 250)
        obs = obs[-self.cropped_rows:, ...]
        obs = obs / 255.
        # obs = np.expand_dims(obs, axis=(0, -1))
        obs = np.expand_dims(obs, axis=0)
        return obs

    def build_Pnet(self):
        image_shape = (self.cropped_rows, self.state_size[1], 3)

        inputs = Input(shape=image_shape)
        # x = Conv2D(32, (3, 3), activation='relu')(inputs)
        # x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)

        mu = Dense(1, activation=None)(x)
        std = Dense(1, activation='softplus')(x)

        model = Model(inputs=inputs, outputs=[mu, std])
        return model

    def build_Qnet(self):
        image_shape = (self.cropped_rows, self.state_size[1], 3)

        state = Input(shape=image_shape)
        action = Input(shape=self.action_size)

        # x = Conv2D(32, (3, 3), activation='relu')(state)
        # x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(state)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        h1 = Dense(64, activation='relu')(x)
        h1 = BatchNormalization()(h1)

        h2 = Dense(16, activation='relu')(action)
        h2 = BatchNormalization()(h2)
        cat = Concatenate()([h1, h2])

        q = Dense(64, activation='relu')(cat)
        q = BatchNormalization()(q)
        q = Dense(1, activation=None)(q)

        model = Model(inputs=[state, action], outputs=q)
        return model

    def load_models(self, pw, q1w, q2w):
        self.pi.load_weights(pw)
        self.q1.load_weights(q1w)
        self.q2.load_weights(q2w)

    # Save the model which is under training
    def save_models(self, pw, q1w, q2w):
        self.pi.save_weights(pw)
        self.q1.save_weights(q1w)
        self.q2.save_weights(q2w)

    def summary(self):
        self.pi.summary()
        self.q1.summary()
        self.q2.summary()

    def hard_update_target_models(self):
        self.q1_target.set_weights(self.q1.get_weights())
        self.q2_target.set_weights(self.q2.get_weights())

    def soft_update_target_models(self):
        q1_params = self.q1.get_weights()
        q1_target_params = self.q1_target.get_weights()
        self.q1_target.set_weights(
            [self.tau * q1w + (1 - self.tau) * q1tw for q1w, q1tw in zip(q1_params, q1_target_params)]
        )

        q2_params = self.q2.get_weights()
        q2_target_params = self.q2_target.get_weights()
        self.q2_target.set_weights(
            [self.tau * q2w + (1 - self.tau) * q2tw for q2w, q2tw in zip(q2_params, q2_target_params)]
        )
        pass

    # Get action from model using epsilon-greedy policy
    def get_action(self, s_t):
        mu, std = self.pi(s_t)
        dist = tfp.distributions.Normal(mu, std)
        action_ = dist.sample()
        log_pi_ = dist.log_prob(action_)

        action = tf.tanh(action_)
        log_pi = log_pi_ - tf.math.log(1 - action**2 + self.epsilon)
        return action, log_pi

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def ready_for_training(self):
        return len(self.memory) >= self.train_start

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return None
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        # construct minibatch inputs
        state_t, action_t, reward_t, state_t1, done = zip(*minibatch)
        state_t = np.concatenate(state_t)
        action_t = np.concatenate(action_t)
        reward_t = np.concatenate(reward_t)
        state_t1 = np.concatenate(state_t1)
        done = np.concatenate(done)

        # train Q-nets
        with tf.GradientTape() as tape1:
            q1 = self.q1([state_t, action_t])
            a1, log_pi_a1 = self.get_action(state_t1)
            q1_target = self.q1_target([state_t1, a1])
            q2_target = self.q2_target([state_t1, a1])
            min_q_target = tf.minimum(q1_target, q2_target)

            soft_q_target = min_q_target - self.alpha * log_pi_a1
            y = tf.stop_gradient(reward_t + self.discount_factor * done * soft_q_target)

            critic1_loss = tf.reduce_mean((q1 - y) ** 2)

        with tf.GradientTape() as tape2:
            q2 = self.q2([state_t, action_t])
            a1, log_pi_a1 = self.get_action(state_t1)
            q1_target = self.q1_target([state_t1, a1])
            q2_target = self.q2_target([state_t1, a1])
            min_q_target = tf.minimum(q1_target, q2_target)

            soft_q_target = min_q_target - self.alpha * log_pi_a1
            y = tf.stop_gradient(reward_t + self.discount_factor * done * soft_q_target)

            critic2_loss = tf.reduce_mean((q2 - y) ** 2)

        grad1 = tape1.gradient(critic1_loss, self.q1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(grad1, self.q1.trainable_variables))

        grad2 = tape2.gradient(critic2_loss, self.q2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(grad2, self.q2.trainable_variables))

        # train P-net
        with tf.GradientTape() as tape3:
            a, log_pi_a = self.get_action(state_t)

            q1 = self.q1([state_t, a])
            q2 = self.q2([state_t, a])
            min_q = tf.minimum(q1, q2)

            actor_loss = tf.reduce_mean(self.alpha * log_pi_a - min_q)

        grad3 = tape3.gradient(actor_loss, self.pi.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grad3, self.pi.trainable_variables))

        # update alpha
        with tf.GradientTape() as tape4:
            a, log_pi_a = self.get_action(state_t)
            alpha_loss = tf.reduce_mean(-self.alpha * (log_pi_a + self.target_entropy))

        grad4 = tape4.gradient(alpha_loss, [self.alpha])
        self.alpha_optimizer.apply_gradients(zip(grad4, [self.alpha]))

        # update target networks
        self.soft_update_target_models()

        return critic1_loss.numpy(), critic2_loss.numpy(), actor_loss.numpy(), alpha_loss.numpy()




class SAC_VAE_Agent(SAC_Agent):

    def __init__(self, state_size, action_size, train=True, z_size=None, vae_weights=None):
        self.z_size = z_size
        self.vae_weights = vae_weights

        super().__init__(state_size, action_size, train=train)

        # Create VAE encoder model
        self.encoder = self.build_vae_encoder()

    def build_vae_encoder(self):
        encoder = Sequential()
        encoder.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu', name='enc_conv1',
                           input_shape=self.state_size))
        encoder.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', name='enc_conv2'))
        encoder.add(Conv2D(128, (4, 4), strides=(2, 2), activation='relu', name='enc_conv3'))
        encoder.add(Conv2D(256, (4, 4), strides=(2, 2), activation='relu', name='enc_conv4'))
        encoder.add(Flatten())  # shape: [-1, 3 * 8 * 256]
        encoder.add(Dense(self.z_size, name='enc_fc_mu'))

        if self.vae_weights:
            encoder.get_layer('enc_conv1').set_weights(self.vae_weights[0:2])
            encoder.get_layer('enc_conv2').set_weights(self.vae_weights[2:4])
            encoder.get_layer('enc_conv3').set_weights(self.vae_weights[4:6])
            encoder.get_layer('enc_conv4').set_weights(self.vae_weights[6:8])
            encoder.get_layer('enc_fc_mu').set_weights(self.vae_weights[8:10])
            encoder.trainable = False
        return encoder

    def preprocess_image(self, obs):
        img_rows, img_cols = self.state_size[0:2]
        obs = cv2.resize(obs, (img_cols, img_rows))
        obs = obs / 255.
        obs -= 0.5
        obs *= 2.
        obs = np.expand_dims(obs, axis=0)   # (1, rows, cols, c)
        encoded = self.encoder.predict(obs)    # (1, z_size)
        return encoded

    def build_Pnet(self):
        inputs = Input(shape=(self.z_size, ))
        x = Dense(128, activation='relu')(inputs)
        mu = Dense(1, activation=None)(x)
        std = Dense(1, activation='softplus')(x)
        model = Model(inputs=inputs, outputs=[mu, std])
        return model

    def build_Qnet(self):
        state = Input(shape=(self.z_size, ))
        action = Input(shape=(self.action_size, ))
        x = Concatenate()([state, action])
        q = Dense(128, activation='relu')(x)
        q = Dense(1, activation=None)(q)
        model = Model(inputs=[state, action], outputs=q)
        return model

    def summary(self):
        super().summary()
        self.encoder.summary()

