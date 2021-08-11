#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
from collections import namedtuple, deque



class ActionOneHot():
    """
    Actions can be stored in a vector of form
    0   (binary) 1 for shooting 0 otherwise
    1   (binary) 1 for thrusting 0 otherwise
    2   (binary) 1 for changing randomly direction 0 otherwise
    """

    # constant size of an action vector
    size = 3

    def __init__(self, shoot=False, thrust=False, pointing=False, vector=None):
        """
        shoot (bool) True is we want to shoot
        thrust (bool) True is we want to thrust
        pointing (bool) True is we want to change direction
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
            self.toVector()

    def toVector(self):
        vector = np.zeros((ActionOneHot.size, 1), dtype=int)
        # value 1 means shoot = True
        vector[0] = int(self.shoot)
        # value 1 means thrust = True
        vector[1] = int(self.thrust)
        # value 1 means pointing = True
        vector[2] = int(self.pointing)
        # squeeze will put it horizontally
        # when using my "from scratch" network I used it vertically for convenience of readability
        # self.vector = vector.squeeze()
        print("shape ", self.vector.shape)
        return self.vector


    def fromVector(self, vector):
        # print(vector)
        if vector.size != ActionOneHot.size:
            raise Exception("Invalid vector : expected size {} but got size {}.".format(ActionOneHot.size, vector.size))

        self.shoot = bool(vector[0])
        self.thrust = bool(vector[1])
        self.pointing = bool(vector[2])


    def possible_actions(self):
        """The ship has a very good computer.
        The ship know what are his possibilities."""
        raise Exception("Not implemented.")


    def __str__(self):
        return "ActionOneHot({0})".format(str(list(self.vector)))

    def __repr__(self):
        return "ActionOneHot({0})".format(str(list(self.vector)))