# sources
# https://cdancette.fr/2018/01/03/reinforcement-learning-part3/

import numpy as np

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam#, sgd
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import (Input, Concatenate, Conv2D, Flatten,
                          MaxPooling2D, BatchNormalization, Reshape,
                          UpSampling2D)
from keras.models import Model

import random
import os
from math import log

import tensorflow as tf

from collections import deque
import time
import datetime
# from IPython.core.debugger import set_trace

from ofighters.lib.action import Action
from ofighters.lib.one_hot_action import ActionOneHot
from ofighters.lib.couple import Point
from ofighters.agents.agent import Agent
from ofighters.lib.observation import DEFAULT_WIDTH, DEFAULT_HEIGHT
from ofighters.lib.epsilon import Epsilon_cos, Epsilon_decay
from ofighters.lib.utils import now

NETWORKS_FOLDER = "networks"

# flatten = lambda l: [item for sublist in l for item in sublist]


REWARDS = {
    "death" : 0, # -80
    "kill" : 0, # 10
    "aim" : 2,
    "trajectory" : 1,
}

# defining the neural network
class Trainer:
    def __init__(self, name=None, learning_rate=0.001, epsilon=Epsilon_decay(), batch_size=30, memory_size=400):
        self.state_size = 320008 # see observation.py
        self.action_size = 2 # see action.py
        self.gamma = 0.9

        self.epsilon = epsilon
        # self.epsilon_min = 0.01
        # self.epsilon_decay = epsilon_decay

        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        self.name = name

        # to be ploted to see inside the brain of mr ship
        self.ptr_values = None
        self.act_values = None

        architecture = "pointer_model"

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
            # TODO give names
            # TODO search for kernel_initializer='he_uniform'
            # Define two input layers
            image_input = Input((DEFAULT_WIDTH, DEFAULT_HEIGHT, 2))
            vector_input = Input((8,))

            # Convolution 400x400 -> 200x200
            conv_layer1 = Conv2D(8, (3,3), padding='same')(image_input)
            norm1 = BatchNormalization()(conv_layer1)
            relu1 = Activation("relu")(norm1)
            pool_layer1 = MaxPooling2D(pool_size=(2, 2))(relu1)
            # Convolution 200x200 -> 100x100
            conv_layer2 = Conv2D(8, (3,3), padding='same')(pool_layer1)
            norm2 = BatchNormalization()(conv_layer2)
            relu2 = Activation("relu")(norm2)
            pool_layer2 = MaxPooling2D(pool_size=(2, 2))(relu2)
            # Convolution 100x100 -> 50x50
            conv_layer3 = Conv2D(8, (3,3), padding='same')(pool_layer2)
            norm3 = BatchNormalization()(conv_layer3)
            relu3 = Activation("relu")(norm3)
            pool_layer3 = MaxPooling2D(pool_size=(2, 2))(relu3)
            # Convolution 50x50 -> 25x25
            conv_layer4 = Conv2D(8, (3,3), padding='same')(pool_layer3)
            norm4 = BatchNormalization()(conv_layer4)
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

        elif architecture == "pointer_model":
            # Define two input layers
            image_input = Input((DEFAULT_WIDTH, DEFAULT_HEIGHT, 2))
            vector_input = Input((8,))

            # Convolution 400x400 -> 200x200
            conv_layer1 = Conv2D(8, (3,3), padding='same', kernel_initializer='he_uniform')(image_input)
            norm1 = BatchNormalization()(conv_layer1)
            relu1 = Activation("relu")(norm1)
            pool_layer1 = MaxPooling2D(pool_size=(2, 2))(relu1)
            # Convolution 200x200 -> 100x100
            conv_layer2 = Conv2D(8, (3,3), padding='same', kernel_initializer='he_uniform')(pool_layer1)
            norm2 = BatchNormalization()(conv_layer2)
            relu2 = Activation("relu")(norm2)
            pool_layer2 = MaxPooling2D(pool_size=(2, 2))(relu2)
            # Convolution 100x100 -> 50x50
            conv_layer3 = Conv2D(8, (3,3), padding='same', kernel_initializer='he_uniform')(pool_layer2)
            norm3 = BatchNormalization()(conv_layer3)
            relu3 = Activation("relu")(norm3)
            pool_layer3 = MaxPooling2D(pool_size=(2, 2))(relu3)
            # Convolution 50x50 -> 25x25
            conv_layer4 = Conv2D(8, (3,3), padding='same', kernel_initializer='he_uniform')(pool_layer3)
            norm4 = BatchNormalization()(conv_layer4)
            relu4 = Activation("relu")(norm4)
            pool_layer4 = MaxPooling2D(pool_size=(2, 2))(relu4)

            # Flatten for the image
            flat_layer = Flatten()(pool_layer4)
            flatten_size = 8 * 25 * 25

            # Concatenate the convolutional features and the vector input
            concat_layer= Concatenate()([vector_input, flat_layer])
            dense1 = Dense(100, activation='relu')(concat_layer)

            # output 1 (shoot or thrust)
            dense2 = Dense(50, activation='relu')(dense1)
            out_1_action_size = 2
            output1 = Dense(out_1_action_size, activation='linear', name='output1')(dense2)

            # output 2 (heatmap of where to point)
            updense1 = Dense(1*25*25, activation='relu')(dense1)
            reshape1 = Reshape((25, 25, 1))(updense1)
            # Convolution 25x25 -> 50x50
            upsampling1 = UpSampling2D((2, 2), interpolation='bilinear')(reshape1)
            upconv_layer1 = Conv2D(2, (3,3), padding='same', kernel_initializer='he_uniform')(upsampling1)
            upnorm1 = BatchNormalization()(upconv_layer1)
            uprelu1 = Activation("relu")(upnorm1)

            # Convolution 50x50 -> 100x100
            upsampling2 = UpSampling2D((2, 2), interpolation='bilinear')(uprelu1)
            upconv_layer2 = Conv2D(4, (3,3), padding='same', kernel_initializer='he_uniform')(upsampling2)
            upnorm2 = BatchNormalization()(upconv_layer2)
            uprelu2 = Activation("relu")(upnorm2)

            # Convolution 100x100 -> 200x200
            upsampling3 = UpSampling2D((2, 2), interpolation='bilinear')(uprelu2)
            upconv_layer3 = Conv2D(8, (3,3), padding='same', kernel_initializer='he_uniform')(upsampling3)
            upnorm3 = BatchNormalization()(upconv_layer3)
            uprelu3 = Activation("relu")(upnorm3)

            # Convolution 200x200 -> 400x400
            upsampling4 = UpSampling2D((2, 2), interpolation='bilinear')(uprelu3)
            upconv_layer4 = Conv2D(1, (3,3), padding='same', kernel_initializer='he_uniform')(upsampling4)
            output2 = Activation("linear")(upconv_layer4)

            # define a model with a list of two inputs and 2 outputs
            model = Model(inputs=[image_input, vector_input], outputs=[output1, output2])
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        model.summary()
        self.model = model
        
    def decay_epsilon(self):
        self.epsilon.next()
        # self.epsilon *= self.epsilon_decay
    
    def get_best_action(self, obs, rand=True):

        if rand and np.random.rand() <= self.epsilon.get():
            # The agent acts randomly
            # return random.randrange(self.action_size)
            return random_play()

        # Predict the reward value based on the given state
        # act_values = self.model.predict(obs.vector.T)
        img_input = np.expand_dims(np.stack((obs.ship_map, obs.laser_map), axis=2), axis=0)
        # act_values = self.model.predict([img_input, obs.vector[:8].T], workers=8, use_multiprocessing=True)[0]
        [act_values_0, ptr_values_0] = self.model.predict([img_input, obs.vector[:8].T], workers=8, use_multiprocessing=True)
        # print("ptr_values_0.shape", ptr_values_0.shape)
        # [0] to extract batch dim
        # squeeze to remove filter dim
        self.act_values = act_values_0[0]
        self.ptr_values = ptr_values_0[0].squeeze()

        # Pick the action based on the predicted reward
        iaction = np.argmax(self.act_values)
        # print("np.argmax(ptr_values, axis=None)", np.argmax(ptr_values, axis=None))
        ipointer = np.unravel_index(np.argmax(self.ptr_values, axis=None), self.ptr_values.shape[0:2], order='F')
        # print("ptr_values[ipointer]", ptr_values[ipointer])
        # print("ptr_values.shape", ptr_values.shape)
        # print("ipointer", ipointer)
        # print("type ipointer", type(ipointer))
        #
        # print("prediction", act_values)
        # ACTION = ["shoot", "thrust", "turn"]
        # print("action", ACTION[iaction])
        # print("pointer", ipointer)

        # act_vector = np.zeros((ActionOneHot.size, 1))
        # act_vector[iaction] = 1
        # print("act vector", act_vector.shape)
        # action = ActionOneHot(vector=act_vector)
        return [iaction, ipointer]

    def remember(self, state, iaction, ipointer, reward, next_state, done):
        self.memory.append([state, iaction, ipointer, reward, next_state, done])

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))

        minibatch = random.sample(self.memory, batch_size)

        # inputs = np.zeros((batch_size, self.state_size))
        inputs1 = np.zeros((batch_size, DEFAULT_WIDTH, DEFAULT_HEIGHT, 2))
        inputs2 = np.zeros((batch_size, 8))
        outputs1 = np.zeros((batch_size, self.action_size))
        outputs2 = np.zeros((batch_size, DEFAULT_WIDTH, DEFAULT_HEIGHT, 1))

        for i, (obs, iaction, ipointer, reward, next_obs, done) in enumerate(minibatch):
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

            # target = self.model.predict([img_input, obs.vector[:8].T], workers=8, use_multiprocessing=True)[0]
            [target, ptr_target] = self.model.predict([img_input, obs.vector[:8].T], workers=8, use_multiprocessing=True)
            # remove batch dimension
            target, ptr_target = target[0], ptr_target[0]
            # print("target", target)
            # print("ptr_target", ptr_target.shape)
            # print("target", target)
            # prediction = self.model.predict(next_obs.vector.T)
            img_input = np.expand_dims(np.stack((next_obs.ship_map, next_obs.laser_map), axis=2), axis=0)
            [prediction, ptr_prediction] = self.model.predict([img_input, next_obs.vector[:8].T], workers=8, use_multiprocessing=True)
            # remove batch dimension
            prediction, ptr_prediction = prediction[0], ptr_prediction[0]
            # print("prediction", prediction)
            # second term is 0 if done is true
            target[iaction] = reward + self.gamma * np.max(prediction) * int(not done)
            ptr_target[ipointer] = reward + self.gamma * np.max(ptr_prediction) * int(not done)

            inputs1[i] = img_input
            inputs2[i] = next_obs.vector[:8].T
            outputs1[i] = target
            outputs2[i] = ptr_target
        fited = self.model.fit(x=[inputs1, inputs2], y=[outputs1, outputs2], epochs=1, batch_size=batch_size, workers=8, use_multiprocessing=True) # verbose=0
        return fited

    def save(self, id=None, overwrite=False):
        name = 'keras-model'
        if self.name:
            name += '-' + self.name
        else:
            name += '-' + now()
        if id:
            name += '-' + id

        self.model.save(NETWORKS_FOLDER + "/" + name, overwrite=overwrite)

# decaying epsilon
# MAX_TIME = 200
# DECAY = 0.99990
# print("Time before decaying to 1% :", log(0.01) / log(DECAY) / 200, "episodes." )

lr = 0.0001 # 0.0001
# TRAINER = Trainer(learning_rate=0.001, epsilon_decay=0.999995, batch_size=8)
# 110 episodes period times the number of actions
TRAINER = Trainer(learning_rate=lr, epsilon=Epsilon_cos(period=110*400), batch_size=8,
                  name="bi_head_pointer"
                  )


def smooth(vector, width=30):
    return np.convolve(vector, [1/width]*width, mode='valid')


def random_play():
    iaction = random.randint(0, TRAINER.action_size-1)
    # TODO tuple ?
    ipointer = (random.randint(0, DEFAULT_WIDTH-1), random.randint(0, DEFAULT_HEIGHT-1))
    return [iaction, ipointer]


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
        self.losses = []
        self.epsilons = []
        # self.exploration = 0.1
        self.done = False
        self.is_learning = True

        self.trainer = TRAINER

        self.previous_obs = None
        self.previous_action = None
        self.previous_pointer = None

        self.collecting_steps = 20
        # save model every N episodes
        self.snapshot = 50

        # the neural network that can controls the ship actions
        # self.model =

        # set model to evaluation mode
        # self.model.eval()


    def reset(self):
        # print("BOT RESET")
        super().reset()
        # All bots share the same trainer so we only save it once
        if self.id == 1:
            self.epsilons.append(self.trainer.epsilon.get())
        self.done = False
        # we restart so we forget the previous action
        self.previous_obs = None
        self.previous_action = None
        self.previous_pointer = None


    def play(self, obs):
        # play will be called even after the ship death so he can observe if he wants
        # but here we only remember the last losing frame and we don't care after
        # and just give anything (that will not be played anyway)
        if self.done:
            return None

        if obs.done:
            if self.is_learning:
                l = self.trainer.replay(self.batch_size)
                self.losses.append(l.history['loss'][0])
                if self.episode % 1 == 0:
                    print("episode: {}, moves: {}, score: {}, epsilon: {}, loss: {}"
                          .format(self.episode, self.steps, self.score, self.trainer.epsilon.get(), self.losses[-1]))
            self.done = True

        if self.previous_obs is not None and self.previous_action is not None and self.previous_pointer is not None:
            # TODO maybe we can only save obs ? better ?
            self.trainer.remember(self.previous_obs, self.previous_action, self.previous_pointer, obs.reward, obs, obs.done)

        # we start with a sequence to collect information (still with learning)
        if (self.total_steps < self.collecting_steps):# or (random.random() < self.exploration) :
            # action = self.collecting_agent.bot_play(obs)
            [iaction, ipointer] = random_play()
        else:
            [iaction, ipointer] = self.trainer.get_best_action(obs)
            # All bots share the same trainer so we only save it once
            if self.is_learning and self.id == 1:
                self.trainer.decay_epsilon()
        self.previous_obs = obs
        self.previous_action = iaction
        self.previous_pointer = ipointer
        # self.previous_action = action.vector

        # All bots share the same trainer so we only save it once
        # and replay it once as remember is also shared
        # print("total steps / 50", self.total_steps / 50)
        if self.is_learning and self.id == 1:
            if self.total_steps % 50 == 0:
                l = self.trainer.replay(self.batch_size)
                self.losses.append(l.history['loss'][0])
                if self.episode % 1 == 0:
                    print("episode: {}, moves: {}, score: {}, epsilon: {}, loss: {}"
                          .format(self.episode, self.steps, self.score, self.trainer.epsilon.get(), self.losses[-1]))
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

        act_vector = np.zeros((Action.size, 1))
        # act_vector[iaction] = 1
        act_vector[iaction] = 1 # concern the first 2 cells
        act_vector[2] = ipointer[0] # coords of the pointer
        act_vector[3] = ipointer[1]
        # action = ActionOneHot(vector=act_vector)
        # print("act_vector", act_vector)
        action = Action(vector=act_vector)
        # print(action)
        return action
