#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math

from random import randint, random, choice
from math import sqrt, acos

from ofighters.lib.form import Circle, sum_angles
from ofighters.lib.couple import Point

from ofighters.lib.laser import Laser
from ofighters.lib.observation import DEFAULT_WIDTH, DEFAULT_HEIGHT
from ofighters.lib.action import Action
from ofighters.lib.one_hot_action import ActionOneHot

from ofighters.agents.agent import Agent

# from brAIn import BrAIn
# from qlearnIA import QlearnIA
from ofighters.agents.qlearnIA_V2 import QlearnIA, REWARDS

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

        self.color = "Yellow"
        self.laser_color = "Red"

        # TODO improve. for the moment the score and steps are reset if the bot change in the middle
        if behavior == "network":
            self.agent = BrAIn()
            # self.initial_agent = BrAIn()
        elif behavior == "QlearnIA":
            self.agent = QlearnIA()
            # self.initial_agent = QlearnIA()
            # self.color = '#AA3300'
            self.color = "Green"
            self.laser_color = "Pink"
        else:
            self.agent = Agent(behavior)
            # self.initial_agent = Agent(behavior)

        # if the ia is in leroy mode time > 0
        self.leroy_time = 0

        # if the ship need to be actualised graphically
        self.actualise = False

        #  the ship is colorful
        self.laser_speed = 10

        self.obs_vector = np.array([])
        self.act_vector = np.array([])


    def reset(self, x=None, y=None):
        # print("ship reset")
        # print(self.agent)
        self.agent.reset()

        self.time = 0
        self.speed = 0
        self.pointing = Point(self.body.x, self.body.y)
        self.body.x = x or self.body.x
        self.body.y = y or self.body.y
        self.state = "flying"
        self.can_shoot = 1

        self.obs_vector = np.array([])
        self.act_vector = np.array([])

    def is_playable(self):
        return self.state not in ["destroyed", "wreckage"]

    def is_super_bot(self):
        return self.agent.behavior in ["network", "QlearnIA"]

    def is_me(self, ship):
        return self.id == ship.id

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

            if self.any_enemy_aimed(self.pointing):
                self.agent.reward += REWARDS["aim"]

            if self.any_enemy_on_trajectory(self.pointing):
                self.agent.reward += REWARDS["trajectory"]

    def any_enemy_aimed(self, pointing):
        found = False
        for ship in self.battleground.ships:
            found = found or not self.is_me(ship) and ship.is_playable() and self.enemy_aimed(pointing, ship)
        if found :
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> AIMED <<<<<")
        return found

    def enemy_aimed(self, pointing, ship):
        # calcluer la ligne pas juste le ciblage
        # score plus élevé si ciblé
        return ship.body.distance(pointing) <= ship.body.radius

    def any_enemy_on_trajectory(self, pointing):
        found = False
        for ship in self.battleground.ships:
            found = found or not self.is_me(ship) and ship.is_playable() and self.enemy_on_trajectory(pointing, ship)
        if found :
            print("------------------------------------------------------------------------------>>>>>>> TRAJECTORY <<<<<")
        return found

    def enemy_on_trajectory(self, pointing, ship):
        # print("pointing", pointing)
        # print("ship.body", ship.body)
        # angle of the shoot relative to this ship [-Pi,Pi]
        # we convert to [0,2Pi]
        shooting_angle = self.body.angle_with(pointing) + math.pi
        if not shooting_angle:
            # print("touch", False)
            return False
        # print("shooting_angle", math.degrees(shooting_angle))
        # relative angle between this ship and the enemy one [-Pi,Pi]
        # we convert to [0,2Pi]
        target_angle = self.body.angle_with(ship.body) + math.pi
        if not target_angle:
            # print("touch", False)
            return False
        # print("target_angle", math.degrees(target_angle))
        # angular radius of the enemy ship [0,2Pi]
        # ie. the relative angular size of the target
        # the biggest and the closest the target is the easiest it is to touch it
        angular_radius = self.body.angular_radius(ship.body)
        # print("angular_radius", math.degrees(angular_radius))
        # 2 angles defining a cone [0,2Pi]
        # you have to aim inside to touch the target
        superior_boundary = sum_angles(target_angle, angular_radius)
        inferior_boundary = sum_angles(target_angle, -angular_radius)
        # print("superior_boundary", math.degrees(superior_boundary))
        # print("inferior_boundary", math.degrees(inferior_boundary))
        # check if the shoot is a touch
        touch = inferior_boundary <= shooting_angle <= superior_boundary
        # print("touch", touch)
        return touch


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
        self.agent.reward += REWARDS["death"]
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

        if obs.done:
            self.agent.step(obs)
            return None

        if self.player:
            action = self.read_keys()
        else:
            action = self.agent.step(obs)

        if not action:
            return None

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
        if not action or not self.is_playable():
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
                # print("action.pointing", action.pointing)
        # print("turn ", self.direction)

        if action.thrust:
            self.thrust()
        if action.shoot:
            self.shoot()

