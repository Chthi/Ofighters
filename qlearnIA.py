
import numpy as np

import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam#, sgd
from keras.layers.advanced_activations import LeakyReLU
import random
import os

from collections import deque
import time
# from IPython.core.debugger import set_trace

from action import Action
from one_hot_action import ActionOneHot
from couple import Point
from agent import Agent


NETWORKS_FOLDER = "ofighter_networks"

flatten = lambda l: [item for sublist in l for item in sublist]

# defining the neural network
class Trainer:
    def __init__(self, name=None, learning_rate=0.001, epsilon_decay=0.9999, batch_size=30, memory_size=400):
        self.state_size = 320008 # see observation.py
        self.action_size = 3 # see action.py
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        self.name = name

        self.graph = tf.get_default_graph()

        if name is not None and os.path.isfile("model-" + name):
            model = load_model("model-" + name)
        else:
            model = Sequential()
            model.add(Dense(400, input_dim=self.state_size, activation='relu', name='input'))
            model.add(Dense(200, input_dim=400, activation='relu', name='dense1'))
            model.add(Dense(100, input_dim=200, activation='relu', name='dense2'))
            model.add(Dense(30, input_dim=100, activation='relu', name='dense3'))
            model.add(Dense(30, input_dim=30, activation='relu', name='dense4'))
            model.add(Dense(self.action_size, input_dim=30, activation='linear', name='output'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        self.model = model
        
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
    
    def get_best_action(self, obs, rand=True):

        if rand and np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)

        with self.graph.as_default():
        # Predict the reward value based on the given state
            act_values = self.model.predict(obs.vector.T)

        # Pick the action based on the predicted reward
        iaction =  np.argmax(act_values[0])

        print("prediction", act_values)
        ACTION = ["shoot", "thrust", "turn"]
        print("action", ACTION[iaction])

        # act_vector = np.zeros((ActionOneHot.size, 1))
        # act_vector[iaction] = 1
        # print("act vector", act_vector.shape)
        # action = ActionOneHot(vector=act_vector)
        return iaction

    def remember(self, state, iaction, reward, next_state):
        self.memory.append([state, iaction, reward, next_state])

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))

        minibatch = random.sample(self.memory, batch_size)

        inputs = np.zeros((batch_size, self.state_size))
        outputs = np.zeros((batch_size, self.action_size))

        with self.graph.as_default():
            for i, (obs_vect, iaction, reward, next_obs_vect) in enumerate(minibatch):
                target = self.model.predict(obs_vect.T)[0]
                # print("target", target)
                prediction = self.model.predict(next_obs_vect.T)
                # print("prediction", prediction)
                target[iaction] = reward + self.gamma * np.max(self.model.predict(next_obs_vect.T))

                inputs[i] = obs_vect.T
                outputs[i] = target
            fited = self.model.fit(inputs, outputs, epochs=1, verbose=0, batch_size=batch_size)
        return fited

    def save(self, id=None, overwrite=False):
        name = 'keras_model'
        if self.name:
            name += '_' + self.name
        else:
            name += '_' + str(time.time())
        if id:
            name += '_' + id
        self.model.save(NETWORKS_FOLDER + "/" + name, overwrite=overwrite)


# TRAINER = Trainer(learning_rate=0.001, epsilon_decay=0.999995, batch_size=8)
TRAINER = Trainer(learning_rate=0.001, epsilon_decay=0.9, batch_size=2)


def smooth(vector, width=30):
    return np.convolve(vector, [1/width]*width, mode='valid')


def random_play(obs):
    """Ship is confused. Ship don't know how to play."""
    act_vector = np.zeros((ActionOneHot.size, 1))
    act_vector[random.randint(0,ActionOneHot.size-1)] = 1
    print("act vector", act_vector.shape)
    action = ActionOneHot(vector=act_vector)
    return action


class QlearnIA(Agent):
    """Ship is smart. Ship can use brain."""

    def __init__(self):
        # we define our own bot called QlearnIA with a behavior define by the function play
        super().__init__(behavior="QlearnIA", bot=self)

        # an agent that can play for the network if it need content or initialisation
        self.collecting_agent = Agent("random")
        self.batch_size = 32
        self.losses = [0]
        self.epsilons = []
        # self.exploration = 0.1

        self.trainer = TRAINER

        self.previous_obs = None
        self.previous_action = None

        self.collecting_steps = 50
        # save model every N episodes
        self.snapshot = 10

        # the neural network that can controls the ship actions
        # self.model =

        # set model to evaluation mode
        # self.model.eval()


    def reset(self):
        self.epsilons.append(self.trainer.epsilon)


    def play(self, obs):
        if self.previous_obs is not None and self.previous_action is not None:
            self.trainer.remember(self.previous_obs.vector, self.previous_action, obs.reward, obs.vector)

        # we start with a sequence to collect information (still with learning)
        if (self.total_steps < self.collecting_steps):# or (random.random() < self.exploration) :
            # action = self.collecting_agent.bot_play(obs)
            iaction = random.randint(0, self.trainer.action_size-1)
        else:
            iaction = self.trainer.get_best_action(obs)
            self.trainer.decay_epsilon()
        self.previous_obs = obs
        self.previous_action = iaction
        # self.previous_action = action.vector

        if self.total_steps % 50 == 0:
            l = self.trainer.replay(self.batch_size)
            self.losses.append(l.history['loss'][0])
            if self.episode % 10 == 0:
                print("episode: {}, moves: {}, score: {}, epsilon: {}, loss: {}"
                      .format(self.episode, self.steps, self.score, self.trainer.epsilon, self.losses[-1]))
        if self.episode > 0 and self.episode % self.snapshot == 0:
            self.trainer.save(id='iteration-%s' % self.episode)

        # reward the networks with the last reward get
        # self.network.update_network_with_reward(self.reward)
        # print(self.network.weights_layers[-1])

        # reinforcement method
        # print("layers", self.network.layers)
        # act_vector = self.network.take_action(obs.vector)

        # TMP as it is the part 1 model
        # just to make sure all is fine for now
        # we only use the linear information of the observation (not the images)
        # with torch.no_grad():
        #     linear_obs = torch.tensor(obs.vector[0:8]).squeeze().float() #.cuda()
        #     out = self.model(linear_obs)
        #     out = np.array(out)
        #     # out = np.array(out.cpu())
        #     act_vector = np.array([
        #         out[0] > 0.5,
        #         out[1] > 0.5,
        #         int(out[2] * 400),
        #         int(out[3] * 400),
        #     ])
            # print(act_vector)

        # print(self.act_vector)
        # print("act_vector shape\n", self.act_vector.shape)
        # print("act_vector\n", self.act_vector)

        act_vector = np.zeros((ActionOneHot.size, 1))
        act_vector[iaction] = 1
        action = ActionOneHot(vector=act_vector)
        return action
