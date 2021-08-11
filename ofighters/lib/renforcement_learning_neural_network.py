

import numpy as np
import math
from collections import namedtuple, deque
from random import random, randint

from ofighters.agents.neural_network import Neural_network

NEGLIGABLE = 0.0005


class Renforcement_learning_neural_network(Neural_network):
    
    def __init__(self, layers, weights_layers=[], biases_layers=[], history_size=100, decay=None):
        super().__init__(layers, weights_layers=[], biases_layers=[])
        # all actions that have been take since the last clear
        # generally the end of a game
        self.actions_history = {"obs" : deque(), "res" : deque()}
        self.max_history_size = history_size
        # since decay^distance = importance
        #       decay = e (log(importance) / distance)
        self.decay = decay or math.exp(math.log(NEGLIGABLE) / self.max_history_size)


    def clear_history(self):
        self.actions_history = {"obs" : deque(), "res" : deque()}


    def take_action(self, observation, exploration_rate=0):
        # there is a chance the action do not follow the network but is random
        if random() < exploration_rate:
            output = np.zeros((self.layers[-1], 1))
            output[randint(0, self.layers[-1]-1)]
            print("exploration")
        else:
            # output of the neural network
            output = self.feed(observation)
        self.actions_history["obs"].append(observation)
        # index of the output vector chose as action to take
        # print("output\n", output)
        # print("type output\n", type(output))
        iaction = np.argmax(output)
        # vector containing the action taken with the "conviction"
        # of the network to choose it. The higher the more the network was sure.
        action = np.zeros(output.shape)
        action[iaction] = output[iaction]
        self.actions_history["res"].append(action)
        # if the maximum size is reached we remove the oldest informations
        if len(self.actions_history["res"]) > self.max_history_size:
            self.actions_history["obs"].popleft()
            self.actions_history["res"].popleft()
        # action vector representing the action taken
        vector = np.zeros(output.shape)
        vector[iaction] = 1
        # print(len(self.actions_history["res"]))
        # print(self.id)
        # print("vector\n", vector)
        # print("action history\n", self.actions_history)
        return vector



    def update_network_with_reward(self, reward):
        """Update the neural network on a mini set using backpropagation/gradient descent.
        reward is the potential reward for the last action
        reward is between -1 and 1"""

        if reward != 0:

            total_decay = self.decay
            # starting from the end
            # print("len", len(self.actions_history["obs"]))
            for i in range(len(self.actions_history["obs"])-1, -1, -1):
                # print(i)
                distances = np.multiply(self.actions_history["res"][i], total_decay * reward)
                (grad_w, grad_b) = self.backprop(self.actions_history["obs"][i], distances)

                # print("i", i)
                # print("obs\n", self.actions_history["obs"][i])
                # print("grad_w\n", grad_w)

                for i in range(self.nb_weights_matrix):
                    # adjusting the values with the average gradient computed using the reward
                    self.weights_layers[i] = self.weights_layers[i] + grad_w[i]
                    self.biases_layers[i] = self.biases_layers[i] + grad_b[i]

                total_decay *= self.decay
