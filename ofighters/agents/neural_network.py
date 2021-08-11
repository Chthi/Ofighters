#!/usr/bin/python

"""
Implement a neural network class

Description :
    A neural network able to
    - train himself on datasets using gradient descent
    - mutate its values with a given percentage
    - be saved in binary files
Author : Thibault Charmet
Date : 04/2018

Notes : 
    - we use neurones containing values between 0 and 1
        it represent the level of activation
    - we use weight and biases values between 0 and 1
        this way it's easy to differenciate good connections
        from bad ones
    - we use matrix "vertically" for reading purpose.
        That mean a layer looks like :
        >>> a = np.zeros((3, 1))
        >>> a
        array([[ 0.],
               [ 0.],
               [ 0.]])

source : youtube.com/watch?v=aircAruvnKk
"""


import pickle
import numpy as np
import os.path
from time import time
from copy import copy, deepcopy


#    -- squashing functions --


def reLU(vector):
    """recrified linear unit function
    R -> [0,+]
    """
    vector[vector<0] = 0
    return vector

def sigmoid(vector):
    """sigmoid function. logistic function.
    R -> [0,1]
    """
    return np.divide(1.0, (1.0 + np.exp(-vector)))

def sigmoid_prime(vector):
    """derivated sigmoid function
    """
    a = sigmoid(vector)
    return np.multiply(a, (1 - a))


# def mzeros(dim):
#     return np.matrix(np.zeros(dim))

# def mones(dim):
#     return np.matrix(np.ones(dim))


# TODO increasing layers size
class Neural_network():
    """
    layers (list) : lengths of the differents layers of the neural network
        size of layers > 1
    """
    id_max = 1

    def __init__(self, layers, weights_layers=[], biases_layers=[]):
        """
        layers (list) : lengths of the differents layers of the neural network
            size of layers > 1
        """
        self.id = Neural_network.id_max
        Neural_network.id_max += 1
        # size of the observation (input) layer
        self.size_obs = layers[0]
        # size of the result (output) layer
        self.size_sol = layers[-1]
        # all the sizes of the vectors constituing the layers of the network
        self.layers = layers
        # number of layers
        self.nb_layers = len(layers)
        self.nb_weights_matrix = self.nb_layers - 1

        # represent an unique string that change with the network caracterisis
        # can be used to check if a network is compatible with a observation/solution vector
        self.network_caracterisis_string = "{}".format("-".join(str(x) for x in self.layers))
        # represent the progression of the current task (in %), generally training
        self.progression = 0
        # number of backpropagations done since the initialisation
        # only informative as it's doesn't 
        self.backpropagations = 0
        # TODO test if shapes corresponds

        # uniform random initialisation of weights and biases between -1 and 1
        self.weights_layers = weights_layers
        self.biases_layers = biases_layers

        if self.weights_layers == []:
            self.weights_layers = [2 * np.random.random((layers[i+1], layers[i])) - 1 for i in range(self.nb_weights_matrix)]
        if self.biases_layers == []:
            self.biases_layers = [2 * np.random.random((layers[i+1], 1)) - 1 for i in range(self.nb_weights_matrix)]

        # display
        for i in range(self.nb_weights_matrix):
            pass
            # print("SHAPE")
            # print(np.array(self.weights_layers[i]).shape)
            # print(np.array(self.biases_layers[i]).shape)
            # print("VALUES")
            # print(np.array(self.weights_layers[i]))
            # print(np.array(self.biases_layers[i]))


    @classmethod    
    def squashing_function(cls, vector):
        return sigmoid(vector)


    @classmethod
    def squashing_function_prime(cls, vector):
        return sigmoid_prime(vector)


    def dist_proto(self, prediction, real):
        """Squared distance between a prediction and a real solution vector"""
        dist = np.zeros(prediction.shape)
        for i in range(0, self.size_sol):
            # print(prediction[i])
            # print(real[i])
            dist[i] = (prediction[i] - real[i]) ** 2
        # print("DIST")
        # print(dist)
        return dist


    def dist(self, prediction, real):
        """Squared distance between a prediction and a real solution"""
        dist = 0
        for i in range(0, self.size_sol):
            dist += (prediction[i] - real[i]) ** 2
        return dist


    def dist_error(self, dataset):
        """Error ratio for all the dataset."""
        n = len(dataset["obs"])
        dist_sum = 0
        err = 0
        for i in range(n):
            prediction = self.feed(dataset["obs"][i])
            # we bring them bask to 0, 1
            dist_sum += np.sum(self.dist(prediction, dataset["sol"][i]))
            if Neural_network.max_sol_index(dataset["sol"][i]) != Neural_network.max_sol_index(prediction):
                err += 1
        mean_dist = dist_sum / n / prediction.shape[0]
        error_ratio = err / n
        print((mean_dist, error_ratio))
        return (mean_dist, error_ratio)


    def error(self, dataset):
        return self.dist_error(dataset)[1]


    def mean_dist(self, dataset):
        return self.dist_error(dataset)[0]

    # TODO
    # error return the % of errors
    # mean_dist the average distance between samples

    def error_old(self, dataset):
        """Error ratio for all the dataset. Between 0 and 1."""
        n = len(dataset["obs"])
        dist_sum = 0
        for i in range(n):
            prediction = self.feed(dataset["obs"][i])
            dist_sum += self.dist(prediction, dataset["sol"][i])
        # print((dist_sum / n)[0, 0])
        return (dist_sum / n)[0, 0]


    def train(self, dataset, nb=None, seuil=None, temps=None, disp=True):
        """Train the neural network on a dataset using backpropagation/gradient descent."""
        
        # by default we train during 5 seconds
        if not (nb or seuil or temps):
            temps = 5
        # TODO : add possibility to train on a limited part of the dataset and calculate error on the other
        # the size of the miniset couldn't be greater than the total dataset
        # size_mini_set = min(size_mini_set, len(dataset["obs"]))

        # begining of th training
        start_time = time()
        # cursor to keep track of the progression in time
        loading = start_time
        # starting error rate
        err = self.error(dataset)
        i = 0
        if disp:
            print("Starting at an error of {}%.".format(err))

        # number of iterations between each print
        if nb:
            spacing = max(nb // 30, 1)
        else:
            spacing = 500

        # stop the training when one of the criteria is reached
        while ((nb != None) and (i != nb)) or ((seuil != None) and (err > seuil)) or ((temps != None) and (loading - start_time < temps)) : 
            # selecting a subset
            self.update_network(dataset["obs"], dataset["sol"])
            loading = time()
            err = self.error(dataset)
            i += 1
            if i % spacing == 0:
                if disp:
                    print("After {} iterations in {} seconds there is {}% of error.".format(i, loading-start_time, err))
        if disp:
            print("trained in {} seconds.".format(loading-start_time))

        # print("Final weights")
        # for weights in self.weights_layers:
        #     print(weights)

        # print("Final biases")
        # for biases in self.biases_layers:
        #     print(biases)


    def update_network(self, observations, solutions):
        """Update the neural network on a mini set using backpropagation/gradient descent.
        solutions or reward must be specified"""

        if len(observations) < 1:
            raise Exception("Observations must contains at least 1 element.")
        if len(solutions) < 1:
            raise Exception("Solutions must contains at least 1 element.")

        # initialisation of weights and biases shaped matrix filled with zeros
        grad_w_sum = [np.zeros(weights.shape) for weights in self.weights_layers]
        grad_b_sum = [np.zeros(biases.shape) for biases in self.biases_layers]

        # print("grad_w_sum shape")
        # for weights in grad_w_sum:
        #     print(weights.shape)

        # summing gradient for all observations in the dataset
        for x in range(0, len(observations)):
            # distances = solutions[x] - self.feed(observations[x])
            distances = solutions[x] - self.feed(observations[x])
            (grad_w, grad_b) = self.backprop(observations[x], distances)

            # print()
            # print("with input ", observations[x])
            # print("self.feed(observations[x])")
            # print(self.feed(observations[x]))
            # print("solutions[x]")
            # print(solutions[x])
            # print("distances")
            # print(distances)
            # for i in range(self.nb_weights_matrix):
            #     print("partiel layer " + str(i))
            #     print(grad_w[i])

            # print("grad_w")
            # print(grad_w)
            # print("grad_w_sum")
            # print(grad_w_sum)

            # grad_w_sum = sum(grad_w_sum, grad_w)
            # grad_b_sum = sum(grad_b_sum, grad_b)

            for i in range(self.nb_weights_matrix):
                grad_w_sum[i] = grad_w_sum[i] + grad_w[i]
                grad_b_sum[i] = grad_b_sum[i] + grad_b[i]

            self.progression = (x+1) / len(observations)
            # print("progression : ", self.progression)

        # adjusting the values with the average gradient computed on all the observations
        grad_w_avg = [np.zeros(weights.shape) for weights in self.weights_layers]
        grad_b_avg = [np.zeros(biases.shape) for biases in self.biases_layers]
        for i in range(self.nb_weights_matrix):
            grad_w_avg[i] = np.divide(grad_w_sum[i], len(observations))
            grad_b_avg[i] = np.divide(grad_b_sum[i], len(observations))

        for i in range(self.nb_weights_matrix):
            self.weights_layers[i] = self.weights_layers[i] + grad_w_avg[i]
            self.biases_layers[i] = self.biases_layers[i] + grad_b_avg[i]
        print("done")

        # print("summed layer " + str(0))
        # print(grad_w_sum)
        # print("average layer " + str(0))
        # print(grad_w_avg)

        # for i in range(self.nb_weights_matrix):
        #     print("summed layer " + str(i))
        #     print(grad_w_sum[i])
        #     print("average layer " + str(i))
        #     print((np.divide(grad_w_sum[i], len(observations))))
        #     # adjusting the values with the average gradient computed on all the observations
        #     self.weights_layers[i] = self.weights_layers[i] - (np.divide(grad_w_sum[i], len(observations)))
        #     self.biases_layers[i] = self.biases_layers[i] - (np.divide(grad_b_sum[i], len(observations)))



    def update_network_continuous(self, observation, solution, immitating_rate=0.01):
        """Update the neural network based on an observations using backpropagation/gradient descent."""

        distances = solution - self.feed(observation)
        (grad_w, grad_b) = self.backprop(observation, distances)

        grad_w = np.multiply(immitating_rate, grad_w)
        grad_b = np.multiply(immitating_rate, grad_b)

        # print("grad_w\n", grad_w)
        # print("grad_b\n", grad_b)

        for i in range(self.nb_weights_matrix):
            # adjusting the values with the average gradient computed using the reward
            self.weights_layers[i] = self.weights_layers[i] + grad_w[i]
            self.biases_layers[i] = self.biases_layers[i] + grad_b[i]



    def backprop(self, obs, distances):
        """return the gradients of the weights and biases to reduce the error
        sol or reward must be specified
        distance is a vector of the output layer size containing the distances
        between given results and expected results
        """

        # gradients of weights and biases
        grad_w = [np.zeros(weights.shape) for weights in self.weights_layers]
        grad_b = [np.zeros(biases.shape) for biases in self.biases_layers]

        # for i in range(self.nb_weights_matrix):
        #     print("grad_w.shape")
        #     print(grad_w[i].shape)
        #     print("grad_b.shape")
        #     print(grad_b[i].shape)

        # >>> feedforward phase
        # activation levels of the current layer
        activation = obs
        # activation levels of the layers
        activations = [obs]
        # non squashed activation levels of layers
        zs = []
        # filling activation values by feeding the network
        for weights, biases in zip(self.weights_layers, self.biases_layers):
            z = np.dot(weights, activation) + biases
            zs.append(z)
            activation = Neural_network.squashing_function(z)
            activations.append(activation)
        # print("activations : \n"+str(activations))
        # print("zs : \n"+str(zs))
        # >>> backward phase

        delta = np.multiply(distances, Neural_network.squashing_function_prime(zs[-1]))

        # print("delta : \n"+str(delta))
        # print("activations[-2] shape")
        # print(activations[-2].shape)
        # print("activations[-2]")
        # print(activations[-2])
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        grad_b[-1] = delta

        # print("grad_w")
        # print(grad_w[-1])

        for layer in range(2, self.nb_layers):
            delta = np.multiply(np.dot(self.weights_layers[-layer+1].transpose(), delta), Neural_network.squashing_function_prime(zs[-layer]))
            # print("activations[-layer-1] shape")
            # print(activations[-layer-1].shape)
            grad_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
            grad_b[-layer] = delta
        # print("grad w : \n"+str(grad_w))
        self.backpropagations += 1
        return (grad_w, grad_b)


    def feed(self, obs):
        """Feed an observation to the network and return the result vector"""
        for weights, biases in zip(self.weights_layers, self.biases_layers):
            # print("obs shape")
            # print(obs.shape)
            # print("weights shape")
            # print(weights.shape)
            # print("biases shape")
            # print(biases.shape)
            # print("weights dot obs + biases shape")
            # print((np.dot(weights, obs) + biases).shape)

            # print("obs")
            # print(obs)
            # print("weights")
            # print(weights)
            # print("biases")
            # print(biases)
            # print("weights dot obs + biases")
            # print((np.dot(weights, obs) + biases))

            obs = Neural_network.squashing_function(np.dot(weights, obs) + biases)
            # print("squashed of that")
            # print(obs.shape)
        return obs


    @classmethod
    def max_sol_index(cls, sol):
        """Return the index of the solution with the highest final layer activation value.
        That mean that we choose the most activated neurone as the solution holder."""
        # print("raw solution")
        # print(sol)
        return np.argmax(sol)


    def inject(self, obs):
        return Neural_network.max_sol_index(self.feed(obs))


    def evolution(self, evolution_rate):
        """Slightly change the weights and biases values.
        evolution_rate between 0 and 1."""
        if evolution_rate > 0 :
            for i in range(len(self.weights_layers)):
                layer = self.weights_layers[i]
                # print("layer")
                # print(layer)
                proportional_change = (2 * np.random.random(layer.shape) - 1) * evolution_rate
                self.weights_layers[i] = np.multiply(proportional_change ,layer) + layer
            for i in range(len(self.biases_layers)):
                layer = self.biases_layers[i]
                proportional_change = (2 * np.random.random(layer.shape) - 1) * evolution_rate
                self.biases_layers[i] = np.multiply(proportional_change ,layer) + layer
        return self


    def mutate(self, mutation_rate):
        """Randomly change a percentage of the weights and biases values.
        mutation_rate between 0 and 1."""
        if mutation_rate > 0 :
            for layer in self.weights_layers + self.biases_layers:
                selected = np.random.random(layer.shape) < mutation_rate
                n = np.sum(selected)
                layer[selected] = 2 * np.random.random((n)) - 1
        return self


    def reproduce(self, n, evolution_rate=0.05, mutation_rate=0.05):
        """Return a list of copies of the same neural network.
        mutation_rate, evolution_rate : if > 0 return modified copies of network"""
        return [deepcopy(self).evolution(evolution_rate).mutate(mutation_rate) for i in range(n)]


    def dafts(self, n):
        """Return a list of too much mutated (daft) neural networks"""
        return [Neural_network(self.layers) for i in range(n)]


    @classmethod
    def generate(cls, n, layers):
        """return a list of random neural networks"""
        return [Neural_network(layers) for i in range(n)]

    def save(self, name=None):
        """Save the neural network as binary file (.dat)"""
        if not name.endswith(".dat"):
            name += ".dat"
        with open(name, "wb") as file:
            pickle.dump(self, file)


    @classmethod
    def load(self, name=None):
        """Load and return a neural network from a binary file (.dat)"""
        if not name.endswith(".dat"):
            name += ".dat"
        with open(name, "rb") as file:
            neural_network = pickle.load(file)
        return neural_network
