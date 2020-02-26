#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import pickle
import os
from random import randint, random, choice

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

import numpy as np

from form import Circle

from map_menu_struct import *


import matplotlib
# While GTK isn't avail everywhere, we use TkAgg backend to generate png
if sys.platform != "win32" and os.getenv("DISPLAY") is None :
    matplotlib.use("Agg")
else :
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from action import Action
from couple import Point
from agent import Agent


# here you can choose the model that will be used by the bot
# It should be the name of the file containing a pickle dump of a Pytorch nn.Module
NETWORKS_FOLDER = "ofighter_networks"
MODEL = "part1_09-07-19_sIAmple"
# MODEL = "part3_31-07-19_BrAInet"


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(8, 15)
        self.linear2 = nn.Linear(15, 4)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x



class BrAInet(nn.Module):

    def __init__(self):
        super(BrAInet, self).__init__()
        # input are ship map and laser map (400x400)
        # 2 input image channel, 8 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(2, 8, 5)
        self.conv2 = nn.Conv2d(8, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64 * 46 * 46, 1024)  # 46x46 from image dimension
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 24)
        # injection of the second size 8 input (1D input)
        self.fc4 = nn.Linear(32, 4)

    def forward(self, vect, imgs):
        x = imgs
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # self.print_layer(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        combined = torch.cat((x.view(x.size(0), -1),
                              vect.view(vect.size(0), -1)), dim=1)
        # x = torch.cat((x, vect), dim=1)

        combined = torch.sigmoid(self.fc4(combined))
        return combined

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class BrAIn(Agent):
    """Ship is smart. Ship can use brain."""

    def __init__(self):
        # we define our own bot called brAIn with a behavior define by the function brAIn_play
        super().__init__(behavior="brAIn", bot=self.brAIn_play)
        # the neural network that can controls the ship actions

        self.model = torch.load(os.path.join(NETWORKS_FOLDER, MODEL+"_pytorch"), map_location='cpu')

        # with open(os.path.join(NETWORKS_FOLDER, MODEL+"_pickle"), "rb") as file:
        #     self.model = pickle.load(file)
        #     # TODO use torch.save instead

        # set model to evaluation mode
        self.model.eval()



    def brAIn_play(self, obs):
        # reward the networks with the last reward get
        # self.network.update_network_with_reward(self.reward)
        # print(self.network.weights_layers[-1])

        # reinforcement method
        # print("layers", self.network.layers)
        # act_vector = self.network.take_action(obs.vector)

        # TMP as it is the part 1 model
        # just to make sure all is fine for now
        # we only use the linear information of the observation (not the images)
        with torch.no_grad():
            linear_obs = torch.tensor(obs.vector[0:8]).squeeze().float() #.cuda()
            out = self.model(linear_obs)
            out = np.array(out)
            # out = np.array(out.cpu())
            act_vector = np.array([
                out[0] > 0.5,
                out[1] > 0.5,
                int(out[2] * 400),
                int(out[3] * 400),
            ])
            # print(act_vector)

        # print(self.act_vector)
        # print("act_vector shape\n", self.act_vector.shape)
        # print("act_vector\n", self.act_vector)
        action = Action(vector=act_vector)

        return action

