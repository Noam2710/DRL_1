import gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ModifiedTensorBoard import ModifiedTensorBoard

epsilon = 1
number_of_episodes = 5000
max_steps_per_episode = 100

learning_rate = 0.1
gamma = 0.99
decay_epsilon = 0.999
window_size = 100

reward_per_episode = dict()
steps_per_episode = dict()
env = gym.make('FrozenLake-v0')
data_holder = []

Q = np.zeros((env.observation_space.n, env.action_space.n))


def print_tests_in_TensorBoard(path_for_file_or_name_of_file=None, read_from_file=False, data_holder=None):
    if read_from_file:
        data_holder_to_visualize = np.load(path_for_file_or_name_of_file)
        name_of_log_dir = '{}-{}'.format(path_for_file_or_name_of_file.split("/")[3],
                                         path_for_file_or_name_of_file.split("/")[4])
        name_of_log_dir = name_of_log_dir.split('.')[0]
        name_of_log_dir += datetime.now().strftime("-%m-%d-%Y-%H-%M-%S")
    else:
        data_holder_to_visualize = data_holder
        name_of_log_dir = path_for_file_or_name_of_file

    tensorboard = ModifiedTensorBoard(log_dir="logs/{}".format(name_of_log_dir))

    for data_of_episode in data_holder_to_visualize:
        tensorboard.step = data_of_episode[0]
        tensorboard.update_stats(Number_of_steps=data_of_episode[1],
                                 cumulative_reward=data_of_episode[2],
                                 epsilon=data_of_episode[3])


def learning(old_state, new_state, reward, action):
    old_q = Q[old_state, action]
    temp_q = reward + gamma * np.max(Q[new_state, :])
    Q[old_state, action] = Q[old_state, action] + learning_rate * (temp_q-old_q)


def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state, :])


def plot_color_maps(episode):
    "taken from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html"

    states = range(Q.shape[0])
    actions = range(Q.shape[1])

    fig, ax = plt.subplots()
    im = ax.imshow(Q,cmap="YlGn")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(actions)))
    ax.set_yticks(np.arange(len(states)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(actions)
    ax.set_yticklabels(states)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(states)):
        for j in range(len(actions)):
            text = ax.text(j, i, round(Q[i, j], 2),
                           ha="center", va="center", color="black", fontsize=7)

    ax.set_title("Q-function values for episode {}".format(episode+1))
    fig.tight_layout()
    fig.savefig('./Q-function-episode-{}.png'.format(episode+1))

    # plt.show()


for episode in range(number_of_episodes):
    current_state = env.reset()
    step = 0
    cumulative_reward = 0

    while step < max_steps_per_episode:
        action = select_action(current_state)
        new_state, reward, if_finished, _ = env.step(action)
        cumulative_reward = reward + gamma * cumulative_reward
        learning(current_state, new_state, reward, action)
        current_state = new_state
        step += 1

        if if_finished:
            if reward == 0:
                step = 100
            data_holder.append([episode, step, cumulative_reward, epsilon])
            break

    epsilon *= decay_epsilon
    if episode in [499, 1999, 4999]:
        plot_color_maps(episode)


data_holder = np.array(data_holder)
print_tests_in_TensorBoard(path_for_file_or_name_of_file="FrozenLake",data_holder=data_holder)









