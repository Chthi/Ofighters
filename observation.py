#!/usr/bin/python
# -*- coding: utf-8 -*-

from copy import copy, deepcopy

import numpy as np

from map_menu_struct import *

import matplotlib
# While GTK isn't avail everywhere, we use TkAgg backend to generate png
if sys.platform != "win32" and os.getenv("DISPLAY") is None :
    matplotlib.use("Agg")
else :
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm



from couple import Couple, Point

DEFAULT_WIDTH = 400
DEFAULT_HEIGHT = 400


class Observation():
    """
     Observations can be stored in a vector of form
     0   (binary) 1 if the ship can shot 0 otherwise
     1   (int) reward from the last frame
     2-3 (int>=0) cell on the map toward the ship is pointing
     4-5 (int>=0) maxX, maxY. Dimensions of the map
     6-7 (int>=0) position of the ship on the map
     w*h (binary 2d array) presence or absence of ship
     w*h (binary 2d array) presence or absence of laser
     """

    observations = {
        "can_shoot" : 1,
        "reward" : 1,
        "pointing" : 2,
        "dim" : 2,
        "pos" : 2,
        "ships_map" : DEFAULT_WIDTH * DEFAULT_HEIGHT,
        "lasers_map" : DEFAULT_WIDTH * DEFAULT_HEIGHT,
    }

    size = sum(observations.values())
    print("size observations", size)


    def __init__(self, **kwargs):
        """
        reward (int) value of the last frame reward
        can_shoot (bool) True if we can shoot
        pointing (Point) point on the map we are pointing
        dim (Couple) pair of integers storing the size of the map
        pos (Point) pair of integers storing the position of the ship
        ships_map (2d array/list like) presence or absence of ship
        lasers_map (2d array/list like) presence or absence of laser
        """
        # vector storing the observation
        self.vector = None

        # are ship and battleground analysed
        self.btlgA = False
        self.shipA = False

        # battleground related infos
        self.battleground = None
        self.dim = None
        self.ship_map = None
        self.laser_map = None

        # ship related infos
        self.reward = None
        self.can_shoot = None
        self.pointing = None
        self.pos = None
        self.done = None

        battleground = kwargs.get('battleground')
        ship = kwargs.get('ship')

        if battleground:
            self.analyse_battleground(battleground)
        if ship:
            self.analyse_ship(ship)


    def analyse_battleground(self, battleground):
        self.battleground = battleground
        self.dim = battleground.dim

        # TODO check if coords need to be reverses as it is a numpy array and y comes first
        # could pose problems if the map is a rectangle
        # or we can try to create them already transposed
        self.ship_map = np.zeros((self.dim.x, self.dim.y))
        for ship in battleground.ships:
            if ship.is_playable():
                self.ship_map = ship.body.binary_draw(self.ship_map)

        self.laser_map = np.zeros((self.dim.x, self.dim.y))
        for laser in battleground.lasers:
            self.laser_map = laser.body.binary_draw(self.laser_map)

        self.btlgA = True
        # .T to transpose to print it the right way
        # print("ship_map\n", self.ship_map.T)
        # print("laser_map\n", self.laser_map.T)


    def analyse_ship(self, ship):
        """Must be executed after analyse_battleground"""
        self.reward = ship.agent.reward
        self.can_shoot = 0 if ship.can_shoot == 0 else 1
        self.pointing = ship.pointing
        self.pos = Point(ship.body.x, ship.body.y)
        self.shipA = True
        # TODO add done to vector ?
        self.done = not ship.is_playable()
        self.toVector()

    # TODO add information isPlayer
    def toVector(self):
        if not self.btlgA:
            raise Exception("You must execute analyse_battleground first.")
        if not self.shipA:
            raise Exception("You must execute analyse_ship first.")

        vector = np.array(self.reward)
        vector = np.append(vector, self.can_shoot)
        vector = np.append(vector, self.pointing.toArray())
        vector = np.append(vector, self.dim.toArray())
        vector = np.append(vector, self.pos.toArray())
        vector = np.append(vector, self.ship_map)
        vector = np.append(vector, self.laser_map)
        # print(vector)
        # putting the vector vertically instead of horizontally
        vector = vector.reshape(vector.size, 1)
        # print("observations vector\n", vector)
        # print("vector size\n", vector.size)
        # print("expected size\n", Observation.size)
        self.vector = vector
        return vector


    def toNumpy(self):
        if not self.btlgA:
            raise Exception("You must execute analyse_battleground first.")
        if not self.shipA:
            raise Exception("You must execute analyse_ship first.")





    def fromVector(self, vector):
        raise Exception("Not implemented.")


    def toBattleground(self):
        raise Exception("Not implemented.")
