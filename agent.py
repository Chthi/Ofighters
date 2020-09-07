#!/usr/bin/python
# -*- coding: utf-8 -*-

import math

from random import randint, random, choice

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



class Agent():

    def __init__(self, behavior=None, bot=None):
        """
        :param bot: object with a "play" method
        """
        self.score = 0
        self.scores = []
        self.reward = 0
        self.steps = 0
        self.total_steps = 0
        self.episode = 0
        self.behavior = behavior

        self.obs_vector = np.array([])
        self.act_vector = np.array([])

        self.bot = None
        if bot:
            self.bot = bot
            self.bot_play = bot.play
        elif not behavior or behavior == "idle":
            self.bot_play = self.idlebot
        elif behavior == "random":
            self.bot_play = self.random_play
        elif behavior == "turret":
            self.bot_play = self.crazy_turret
        elif behavior == "runner":
            self.bot_play = self.crazy_runner
        elif behavior == "thrust":
            self.bot_play = self.never_back_down
        elif behavior == "shoot":
            self.bot_play = self.mass_shooter
        else:
            raise Exception("You must give a bot in parameter or select an existing behavior.")

        #    advanced bots
        # scared bot
        # cac bot
        # snipe bot
        # dodge bot

    def reset(self):
        # print("AGENT RESET")
        self.steps = 0
        self.scores.append(self.score)
        self.score = 0
        self.episode += 1

    def step(self, obs):
        self.steps += 1
        self.total_steps += 1

        if obs.reward != 0:
            print(self.reward)

        self.score += self.reward
        self.reward = 0

        self.obs_vector = obs.vector
        action = self.bot_play(obs)

        if not action:
            return None

        self.act_vector = action.vector

        # print("obs_vector shape\n", obs_vector.shape)
        # print("obs_vector\n", obs_vector)

        # print("I want to ", action.label)
        # if action.label == "turn":
        #     print(action.turn.direction)

        # vector = action.toVector()
        # print("vector", vector)
        # action.fromVector(vector)
        # print("action", action.label)

        return action


    def idlebot(self, obs):
        """Ship don't like life anymore. Ship don't like its taste."""
        shoot = False
        thrust = False
        pointing = obs.pointing
        return Action(shoot=shoot, thrust=thrust, pointing=pointing)


    def never_back_down(self, obs):
        """Thrust ship, thrust !"""
        shoot = False
        thrust = True
        pointing = obs.pointing
        return Action(shoot=shoot, thrust=thrust, pointing=pointing)


    def mass_shooter(self, obs):
        """Who give him that ?"""
        shoot = True
        thrust = False
        pointing = obs.pointing
        return Action(shoot=shoot, thrust=thrust, pointing=pointing)


    def random_play(self, obs):
        """Ship is confused. Ship don't know how to play."""
        possibles = ["shoot", "thrust", "pointing"]
        action = choice(possibles)
        shoot = action == "shoot"
        thrust = action == "thrust"
        if action == "pointing":
            pointing = Point(randint(0, obs.dim.x), randint(0, obs.dim.y))
        else:
            pointing = obs.pointing
        return Action(shoot=shoot, thrust=thrust, pointing=pointing)


    def crazy_turret(self, obs):
        """Ship don't want to move. Ship only want to kill."""
        shoot = random() < 0.8
        thrust = False
        if random() < 0.3:
            pointing = Point(randint(0, obs.dim.x), randint(0, obs.dim.y))
        else:
            pointing = obs.pointing
        return Action(shoot=shoot, thrust=thrust, pointing=pointing)


    def crazy_runner(self, obs):
        """Ship is a fast boi. Ship can dance to."""
        shoot = False
        thrust = random() < 0.9
        if random() < 0.1:
            pointing = Point(randint(0, obs.dim.x), randint(0, obs.dim.y))
        else:
            pointing = obs.pointing
        return Action(shoot=shoot, thrust=thrust, pointing=pointing)












