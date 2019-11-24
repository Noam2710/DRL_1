
import matplotlib.pyplot as plt
import random, gym, numpy as np
import time
import h5py

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from statistics import mean
from datetime import datetime

MAX_EPISODES = 100000
data_holder = []
LEARNING_RATE = 0.001
MEMORY_LENGTH = 2000
INIT_EXPLORE_RATE = 1.0
BATCH_SIZE = 32
GAMMA = 1
DECAYING_EPSILON = 0.9995
EXPLORATION_MIN = 0.02
window_size = 100
THRESHOLD = 475
NUMBER_OF_LAYERS = 5
best_result = 500
episodes_to_losses = dict()


def get_list_of_average_losses():
    list_of_avg_losses = []
    for episode_index in episodes_to_losses:
        if len(episodes_to_losses[episode_index]) == 0:
            list_of_avg_losses.append(0)
        else:
            list_of_avg_losses.append(mean(episodes_to_losses[episode_index]))

    return list_of_avg_losses


def running_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def print_figures(data_holder,time):
    fig, axs = plt.subplots(4, figsize=[12, 12])
    axs[0].plot(data_holder[:, 0], data_holder[:, 1])
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')

    axs[1].plot(data_holder[:, 0], data_holder[:, 3])
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Running Average')

    axs[2].plot(list(range(len(episodes_to_losses))), get_list_of_average_losses())
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Loss')

    axs[3].plot(data_holder[:, 0], data_holder[:, 2])
    axs[3].set_xlabel('Episode')
    axs[3].set_ylabel('Epsilon')

    fig.savefig('./CartPoleResults/figures/{}hiddenlayers/results-{}-cartpole.png'.format(NUMBER_OF_LAYERS,str(time)))


class DQN:
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space
        self.model = self.init_model()

    def init_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.obs_space,), activation="relu"))

        for layer in range(NUMBER_OF_LAYERS-1):
            model.add(Dense(24, activation="relu"))
            model.add(Dense(24, activation="relu"))

        model.add(Dense(self.act_space, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        return model


class DQN_Owner:
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space
        self.target_model = DQN(obs_space, act_space)
        self.behaviour_model = DQN(obs_space, act_space)
        self.copy_fit_model_weights_to_target()
        self.experience_replay = deque(maxlen=MEMORY_LENGTH)
        self.exploration_rate = INIT_EXPLORE_RATE
        self.steps_without_updates = 0
        self.maximum_epochs_without_updates = 16

    def sample_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.act_space)
        else:
            q_values_of_current_state = self.behaviour_model.model.predict(state)
            return np.argmax(q_values_of_current_state[0])

    def run_experience_replay(self,episode_index):
        states_batch = []
        q_values_batch = []
        if len(self.experience_replay) > BATCH_SIZE:
            current_batch = self.sample_batch()
            for state, action, reward, next_state, finished in current_batch:
                if finished:
                    q = reward
                else:
                    q = reward + GAMMA*np.amax(self.target_model.model.predict(next_state)[0])

                q_values = self.target_model.model.predict(state)
                q_values[0][action] = q
                states_batch.append(state[0])
                q_values_batch.append(q_values[0])

            history = self.behaviour_model.model.fit(np.array(states_batch), np.array(q_values_batch),
                                                     epochs=1,
                                                     verbose=0,
                                                     batch_size=BATCH_SIZE)

            episodes_to_losses[episode_index].append(history.history['loss'][0])
            
            self.decay_epsilon()
            self.update_target_network_if_necessary()

    def add_to_experience_replay(self, state, action, reward, next_state, finished):
        self.experience_replay.append((state, action, reward, next_state, finished))

    def copy_fit_model_weights_to_target(self):
        self.target_model.model.set_weights(self.behaviour_model.model.get_weights())

    def decay_epsilon(self):
        if self.exploration_rate > EXPLORATION_MIN:
            self.exploration_rate *= DECAYING_EPSILON

    def update_target_network_if_necessary(self):
        if self.steps_without_updates == self.maximum_epochs_without_updates:
            self.steps_without_updates = 0
            self.copy_fit_model_weights_to_target()
        else:
            self.steps_without_updates += 1

    def sample_batch(self):
        return random.sample(self.experience_replay, BATCH_SIZE)


def train_agent():
    while True:
        data_holder = []
        env = gym.make("CartPole-v1")
        obs_space = env.observation_space.shape[0]
        act_space = env.action_space.n
        dqn_owner = DQN_Owner(obs_space, act_space)
        steps = []
        global best_result

        for episode_index in range(best_result):
            episodes_to_losses[episode_index] = []
            current_state = env.reset()
            current_state = np.reshape(current_state, [1, obs_space])
            step = 0
            while True:
                step += 1
                action = dqn_owner.sample_action(current_state)
                next_state, reward, finished, _ = env.step(action)
                next_state = np.reshape(next_state, [1, obs_space])
                dqn_owner.add_to_experience_replay(current_state, action, reward, next_state, finished)
                current_state = next_state
                dqn_owner.run_experience_replay(episode_index)

                if finished:
                    steps.append(step)
                    running_average = 0 if window_size > len(steps) else mean(steps[(len(steps)-window_size):])
                    print("Episode: {}, Steps:{}, 100-Average: {}, Epsilon:{} ".format(episode_index, step,
                                                                                                    str(running_average),
                                                                                                    str(dqn_owner.exploration_rate,
                                                                                                    )))
                    data_holder.append([episode_index, step, dqn_owner.exploration_rate, running_average])
                    break

            if running_average > THRESHOLD:
                best_result = episode_index
                data_holder = np.array(data_holder)
                time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S-episode-break-{}".format(episode_index))
                np.save('./CartPoleResults/data_holders/{}hiddenlayers/{}.h5'.format(NUMBER_OF_LAYERS,episode_index), data_holder)
                dqn_owner.target_model.model.save('./CartPoleResults/weights/{}hiddenlayers/{}.h5'.format(NUMBER_OF_LAYERS,time))
                print_figures(data_holder, time)
                break


def test_agent():
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space.shape[0]
    act_space = env.action_space.n
    current_state = env.reset()
    current_state = np.reshape(current_state, [1, obs_space])
    target_network = DQN(obs_space,act_space)
    target_network.model = load_model('./CartPoleResults/weights/5hiddenlayers/11-24-2019-08-54-46-episode-break-172.h5')
    step = 0
    while True:
        step +=1
        env.render()
        action = np.argmax(target_network.model.predict(current_state)[0])
        next_state, _, finished, _ = env.step(action)
        next_state = np.reshape(next_state, [1, obs_space])
        current_state = next_state
        if finished:
            print(str(step))
            break



# train_agent()
test_agent()




