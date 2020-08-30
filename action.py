#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
from collections import namedtuple, deque

from map_menu_struct import *

import matplotlib
# While GTK isn't avail everywhere, we use TkAgg backend to generate png
if sys.platform != "win32" and os.getenv("DISPLAY") is None :
    matplotlib.use("Agg")
else :
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from couple import Point



class Action():
    """
    Actions can be stored in a vector of form
    0   (binary) 1 for shooting 0 otherwise
    1   (binary) 1 for thrusting 0 otherwise
    2-3 (int>=0) cell on the map toward the ship will be pointing
    """

    # constant size of an action vector
    size = 4

    def __init__(self, shoot=False, thrust=False, pointing=None, vector=None):
        """
        shoot (bool) True is we want to shoot
        thrust (bool) True is we want to thrust
        pointing (Point) cell on the map toward the ship will point
        vector (array/list like) if given, use it to set all actions attributes
        """
        # vector storing the action
        self.vector = vector
        if vector is not None:
            self.fromVector(vector)
        else:
            self.shoot = shoot
            self.thrust = thrust
            self.pointing = pointing
            if not self.pointing:
                raise Exception("pointing argument must be specified.")
            self.toVector()

    def toVector(self):
        vector = np.zeros((Action.size, 1), dtype=int)
        # value 1 means shoot = True
        vector[0] = int(self.shoot)
        # value 1 means thrust = True
        vector[1] = int(self.thrust)
        # position of the cell we are pointing
        vector[2] = self.pointing.x
        vector[3] = self.pointing.y
        # squeeze will put it horizontally
        # when using my "from scratch" network I used it vertically for convenience of readability
        self.vector = vector.squeeze()
        return vector


    def fromVector(self, vector):
        # print(vector)
        if vector.size != Action.size:
            raise Exception("Invalid vector : expected size {} but got size {}.".format(Action.size, vector.size))

        self.shoot = bool(vector[0])
        self.thrust = bool(vector[1])
        self.pointing = Point(vector[2], vector[3])


    def possible_actions(self):
        """The ship has a very good computer.
        The ship know what are his possibilities."""
        raise Exception("Not implemented.")


    def __str__(self):
        return "Action({0})".format(str(list(self.vector)))

    def __repr__(self):
        return "Action({0})".format(str(list(self.vector)))