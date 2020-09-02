
import numpy as np

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, sgd
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Concatenate, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Model

import random
import os

import tensorflow as tf

from collections import deque
import time
import datetime
# from IPython.core.debugger import set_trace

from action import Action
from one_hot_action import ActionOneHot
from couple import Point
from agent import Agent
from observation import DEFAULT_WIDTH, DEFAULT_HEIGHT

NETWORKS_FOLDER = "ofighter_networks"

flatten = lambda l: [item for sublist in l for item in sublist]

def now():
    return datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")


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

        architecture = "conv2d"

        if name is not None and os.path.isfile("keras-model-" + name):
            model = load_model("keras-model-" + name)

        elif architecture == "dense":
            model = Sequential()
            model.add(Dense(200, input_dim=self.state_size, activation='relu', name='input'))
            model.add(Dense(200, input_dim=200, activation='relu', name='dense1'))
            model.add(Dense(100, input_dim=200, activation='relu', name='dense2'))
            model.add(Dense(30, input_dim=100, activation='relu', name='dense3'))
            model.add(Dense(30, input_dim=30, activation='relu', name='dense4'))
            model.add(Dense(self.action_size, input_dim=30, activation='linear', name='output'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        elif architecture == "conv2d":
            # Define two input layers
            image_input = Input((DEFAULT_WIDTH, DEFAULT_HEIGHT, 2))
            vector_input = Input((8,))

            # Convolution 400x400 -> 199x199
            conv_layer1 = Conv2D(8, (3,3), activation='relu')(image_input)
            norm1 = Dropout(0.2)(conv_layer1)
            relu1 = Activation("relu")(norm1)
            pool_layer1 = MaxPooling2D(pool_size=(2, 2))(relu1)
            # Convolution 199x199 -> 98x98
            conv_layer2 = Conv2D(8, (3,3), activation='relu')(pool_layer1)
            norm2 = Dropout(0.2)(conv_layer2)
            relu2 = Activation("relu")(norm2)
            pool_layer2 = MaxPooling2D(pool_size=(2, 2))(relu2)
            # Convolution 98x98 -> 48x48
            conv_layer3 = Conv2D(8, (3,3), activation='relu')(pool_layer2)
            norm3 = Dropout(0.2)(conv_layer3)
            relu3 = Activation("relu")(norm3)
            pool_layer3 = MaxPooling2D(pool_size=(2, 2))(relu3)
            # Convolution 48x48 -> 23x23
            conv_layer4 = Conv2D(4, (3,3), activation='relu')(pool_layer3)
            norm4 = Dropout(0.2)(conv_layer4)
            relu4 = Activation("relu")(norm4)
            pool_layer4 = MaxPooling2D(pool_size=(2, 2))(relu4)

            # Flatten for the image
            flat_layer = Flatten()(pool_layer4)

            # Concatenate the convolutional features and the vector input
            concat_layer= Concatenate()([vector_input, flat_layer])
            dense1 = Dense(100, activation='relu')(concat_layer)
            dense2 = Dense(50, activation='relu')(dense1)
            output = Dense(self.action_size, activation='linear', name='output')(dense2)

            # define a model with a list of two inputs
            model = Model(inputs=[image_input, vector_input], outputs=output)
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        model.summary()
        self.model = model
        
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
    
    def get_best_action(self, obs, rand=True):

        if rand and np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)

        with self.graph.as_default():
        # Predict the reward value based on the given state
            # act_values = self.model.predict(obs.vector.T)
            img_input = np.expand_dims(np.stack((obs.ship_map, obs.laser_map), axis=2), axis=0)
            act_values = self.model.predict([img_input, obs.vector[:8].T], workers=8, use_multiprocessing=True)[0]

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

        # inputs = np.zeros((batch_size, self.state_size))
        inputs1 = np.zeros((batch_size, DEFAULT_WIDTH, DEFAULT_HEIGHT, 2))
        inputs2 = np.zeros((batch_size, 8))
        outputs = np.zeros((batch_size, self.action_size))

        with self.graph.as_default():
            for i, (obs, iaction, reward, next_obs) in enumerate(minibatch):
                # target = self.model.predict(obs.vector.T)[0]

                img_input = np.expand_dims(np.stack((obs.ship_map, obs.laser_map), axis=2), axis=0)
                # img_input = np.stack((obs.ship_map, obs.laser_map), axis=2)
                # img_input = np.array([obs.ship_map, obs.laser_map])

                # img_input = np.array([obs.ship_map, obs.laser_map])
                # print("obs.vector[:8].T.shape", obs.vector[:8].T.shape)
                # print("img_input", img_input)
                # print("img_input.shape", img_input.shape)
                # print("len(img_input)", len(img_input))
                # print("type(img_input[0])", type(img_input[0]))

                target = self.model.predict([img_input, obs.vector[:8].T], workers=8, use_multiprocessing=True)[0] # workers=8, use_multiprocessing=True
                # print("target", target)
                # prediction = self.model.predict(next_obs.vector.T)
                img_input = np.expand_dims(np.stack((next_obs.ship_map, next_obs.laser_map), axis=2), axis=0)
                prediction = self.model.predict([img_input, next_obs.vector[:8].T], workers=8, use_multiprocessing=True)[0]
                # print("prediction", prediction)
                target[iaction] = reward + self.gamma * np.max(prediction)

                inputs1[i] = img_input
                inputs2[i] = next_obs.vector[:8].T
                outputs[i] = target
            fited = self.model.fit(x=[inputs1, inputs2], y=outputs, epochs=1, batch_size=batch_size, workers=8, use_multiprocessing=True) # verbose=0,
        return fited

    def save(self, id=None, overwrite=False):
        name = 'keras-model'
        if self.name:
            name += '-' + self.name
        else:
            name += '-' + now()
        if id:
            name += '-' + id
        with self.graph.as_default():
            self.model.save(NETWORKS_FOLDER + "/" + name, overwrite=overwrite)


from math import log
DECAY = 0.9995
print("Time before decaying to 1% :", log(0.01) / log(DECAY) )
# TRAINER = Trainer(learning_rate=0.001, epsilon_decay=0.999995, batch_size=8)
TRAINER = Trainer(learning_rate=0.001, epsilon_decay=DECAY, batch_size=8,
                  name="conv2d_dropout"
                  )


def smooth(vector, width=30):
    return np.convolve(vector, [1/width]*width, mode='valid')


def random_play(obs):
    """Ship is confused. Ship don't know how to play."""
    act_vector = np.zeros((ActionOneHot.size, 1))
    act_vector[random.randint(0,ActionOneHot.size-1)] = 1
    # print("act vector", act_vector.shape)
    action = ActionOneHot(vector=act_vector)
    return action


class QlearnIA(Agent):
    """Ship is smart. Ship can use brain."""
    max_id = 1

    def __init__(self):
        # we define our own bot called QlearnIA with a behavior define by the function play
        super().__init__(behavior="QlearnIA", bot=self)

        self.id = QlearnIA.max_id
        QlearnIA.max_id += 1
        # an agent that can play for the network if it need content or initialisation
        self.collecting_agent = Agent("random")
        self.batch_size = 8
        self.losses = [0]
        self.epsilons = []
        # self.exploration = 0.1

        self.trainer = TRAINER

        self.previous_obs = None
        self.previous_action = None

        self.collecting_steps = 20
        # save model every N episodes
        self.snapshot = 10

        # the neural network that can controls the ship actions
        # self.model =

        # set model to evaluation mode
        # self.model.eval()


    def reset(self):
        # print("BOT RESET")
        super().reset()
        # All bots share the same trainer so we only save it once
        if self.id == 1:
            self.epsilons.append(self.trainer.epsilon)


    def play(self, obs):
        if self.previous_obs is not None and self.previous_action is not None:
            self.trainer.remember(self.previous_obs, self.previous_action, obs.reward, obs)

        # we start with a sequence to collect information (still with learning)
        if (self.total_steps < self.collecting_steps):# or (random.random() < self.exploration) :
            # action = self.collecting_agent.bot_play(obs)
            iaction = random.randint(0, self.trainer.action_size-1)
        else:
            iaction = self.trainer.get_best_action(obs)
            # All bots share the same trainer so we only save it once
            if self.id == 1:
                self.trainer.decay_epsilon()
        self.previous_obs = obs
        self.previous_action = iaction
        # self.previous_action = action.vector

        # All bots share the same trainer so we only save it once
        if self.id == 1:
            if self.total_steps % 50 == 0:
                l = self.trainer.replay(self.batch_size)
                self.losses.append(l.history['loss'][0])
                if self.episode % 1 == 0:
                    print("episode: {}, moves: {}, score: {}, epsilon: {}, loss: {}"
                          .format(self.episode, self.steps, self.score, self.trainer.epsilon, self.losses[-1]))
            if self.episode > 0 and self.episode % self.snapshot == 0 and self.steps < 2 :
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
