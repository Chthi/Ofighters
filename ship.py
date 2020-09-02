#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math

from random import randint, random, choice
from math import sqrt, acos

from form import Circle
from couple import Point

from map_menu_struct import *


import matplotlib
# While GTK isn't avail everywhere, we use TkAgg backend to generate png
if sys.platform != "win32" and os.getenv("DISPLAY") is None :
    matplotlib.use("Agg")
else :
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from laser import Laser
from observation import DEFAULT_WIDTH, DEFAULT_HEIGHT
from action import Action
from one_hot_action import ActionOneHot
from agent import Agent
# from brAIn import BrAIn
from qlearnIA import QlearnIA

SHIPS_SPEED = 8
PLAYER_FOLLOWS_NETWORK_RULES = True
NETWORK = True
LEROY_RATE = 0.0
# LEROY_RATE = 0.01



class Ship():
    id_max = 1

    def __init__(self, x, y, battleground, behavior="idle"):

        # id of the fighter
        self.id = Ship.id_max
        Ship.id_max += 1
        self.time = 0
        #  the ship know is own position
        # the ship is not invulnerable. the ship can be hit
        self.body = Circle(x, y, 8)
        # the ship resilience
        self.hull = 1
        # the ship knows in which battleground he is
        self.battleground = battleground
        # the ship know is speed and direction
        self.speed = 0
        self.max_speed = SHIPS_SPEED

        # the ship trajectory is an affine function that goes toward the cell the ship is pointing
        self.pointing = Point(x, y)

        # current state of the ship
        self.state = "flying"
        # there is no cooldown on the laser
        self.can_shoot = 1
        # the human player controlling it. None if no player controls it
        self.player = None
        # TODO improve. for the moment the score and steps are reset if the bot change in the middle
        if behavior == "network":
            self.agent = BrAIn()
            self.initial_agent = BrAIn()
        elif behavior == "q_learning":
            self.agent = QlearnIA()
            self.initial_agent = QlearnIA()
        else:
            self.agent = Agent(behavior)
            self.initial_agent = Agent(behavior)

        # if the ia is in leroy mode time > 0
        self.leroy_time = 0

        # if the ship need to be actualised graphically
        self.actualise = False

        #  the ship is colorful
        # self.color = '#AA3300'
        self.color = "Yellow"
        self.laser_color = "Red"
        self.laser_speed = 10

        self.obs_vector = np.array([])
        self.act_vector = np.array([])


    def reset(self):
        # print("SHIP RESET")
        # print(self.agent)
        self.agent.reset()

        self.time = 0
        self.speed = 0
        self.pointing = Point(self.body.x, self.body.y)
        self.state = "flying"
        self.can_shoot = 1

        self.obs_vector = np.array([])
        self.act_vector = np.array([])


    def is_playable(self):
        return self.state not in ["destroyed", "wreckage"]

    def assign_player(self, player):
        self.player = player
        self.color = "Grey"
        self.laser_color = "Green"  # light green

    def unassign_player(self):
        self.player = None
        self.color = "Yellow"
        self.laser_color = "Red"

    def hit(self, object):
        # TODO : damage by objects, armor, shield etc
        self.hull -= 1
        if self.hull <= 0:
            self.explode()


    def shoot(self):
        edge = self.body.edge(self.pointing.x, self.pointing.y, 2)
        # there is no guarantee that such a edge exists
        if edge:
            laser = Laser(
                edge[0], edge[1],
                self.laser_speed,
                self.pointing, self.battleground,
                owner=self, color=self.laser_color
            )

            # if we point inside the ship hitbox the fired point of the laser become the center of the ship
            # but the real spawn point stay the same
            if self.body.collide(Circle(self.pointing.x, self.pointing.y, 2)):
                laser.fired = Point(self.body.x, self.body.y)

            self.battleground.lasers.append(laser)


    def thrust(self):
        deltaX = self.pointing.x - self.body.x
        deltaY = self.pointing.y - self.body.y
        dist = math.sqrt(deltaX ** 2 + deltaY ** 2)

        if dist != 0:
            dx = deltaX * self.max_speed / dist
            dy = deltaY * self.max_speed / dist
            self.body.x = min(self.battleground.dim.x - 1, max(0, int(self.body.x + dx)))
            self.body.y = min(self.battleground.dim.y - 1, max(0, int(self.body.y + dy)))


    def explode(self):
        # print("baaouummm")
        # dying is bad, remember
        self.agent.reward -= 10
        self.battleground.last_x_time_rewards.append((10, self.battleground.time))
        self.state = "destroyed"


    def read_keys(self):
        shoot = "shoot" in self.player.actions_set
        thrust = "thrust" in self.player.actions_set
        if "pointing" in self.player.actions_set:
            pointing = Point(self.player.cursor.x, self.player.cursor.y)
        else:
            pointing = self.pointing
        self.player.clear_keys()
        # print(shoot, thrust, pointing)
        return Action(shoot=shoot, thrust=thrust, pointing=pointing)


    def go_bottom_reward(self):
        # print(self.body.y)
        # print(self.battleground.dim.y // 2)
        reach_top = not self.on_bottom and self.body.y == self.battleground.dim.y - 1
        self.on_bottom = self.body.y == self.battleground.dim.y - 1
        return int(reach_top)


    def get_action(self, obs):
        """The ship is smart.
        The ship know what to do to be the best."""

        # obs become a state relative to each different ship (could be fog of war or just the position)
        obs.analyse_ship(self)

        if self.player:
            action = self.read_keys()
        else:
            action = self.agent.step(obs)

        if hasattr(self.agent, 'obs_vector') and self.agent.obs_vector.size > 0:
            self.obs_vector = self.agent.obs_vector
        else:
            self.obs_vector = obs.vector

        if hasattr(self.agent, 'act_vector') and self.agent.act_vector.size > 0:
            self.act_vector = self.agent.act_vector
        else:
            self.act_vector = action.vector

        return action


    def leroy_jenkins(self):
        # in leroy mode ship feels reckless
        # crazy people are the one who discovers what others did not even think about
        self.agent = Agent.crazy_runner
        self.leroy_time = randint(10, 100)
        self.color = "Red"
        self.laser_color = "Grey"
        # print("Leeeeeeeeeeeroy Jeeeeeenkins !")
        self.actualise = True


    def back_to_normal(self):
        self.agent = self.initial_agent
        self.color = "Yellow"
        self.laser_color = "Red"
        self.actualise = True


    def move(self, action):
        """The ship can think. The ship can act. The ship is a smart boi."""
        self.time += 1

        # If ship is destroyed ship can only contemplate sadness and despair
        if not self.is_playable():
            return None

        self.actualise = False

        if self.leroy_time == 1:
            self.back_to_normal()
        if self.leroy_time > 0:
            self.leroy_time -= 1

        # there is a chance that the ia enter in leroy mode
        # the ia goes mad for some time, acting randomly
        # added to allow the ships to explore the possible actions and not stay passive
        if not self.player and self.leroy_time == 0 and self.agent.behavior == "network" and random() < LEROY_RATE:
            self.leroy_jenkins()

        # training reward depending on position
        # self.agent.reward = self.go_bottom_reward()

        if isinstance(action, ActionOneHot):
            if action.pointing:
                self.pointing = Point(randint(0, DEFAULT_WIDTH-1), randint(0, DEFAULT_HEIGHT-1))
        elif isinstance(action, Action):
            if action.pointing:
                self.pointing = action.pointing
        # print("turn ", self.direction)

        if action.thrust:
            self.thrust()
        if action.shoot:
            self.shoot()

