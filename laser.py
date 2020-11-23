#!/usr/bin/python
# -*- coding: utf-8 -*-


from math import sqrt

from collections import namedtuple

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

from couple import Point
from qlearnIA_V2 import REWARDS

LIGHT_SPEED = 1 # proportionally change the speed of lasers


class Laser():
    id_max = 1

    def __init__(self, x, y, speed, pointing, battleground, owner=None, color="Red"):
        # print("piouu")
        self.id = Laser.id_max
        Laser.id_max += 1
        self.body = Circle(x, y, radius=2)
        self.speed = speed * LIGHT_SPEED

        # the laser trajectory is an affine function that goes toward the cell the ship is pointing
        self.fired = Point(x, y)
        self.pointing = pointing

        self.battleground = battleground
        self.color = color
        self.time = 0
        self.owner = owner
        self.state = "flying"

    def move(self):
        self.time += 1

        deltaX = self.pointing.x - self.fired.x
        deltaY = self.pointing.y - self.fired.y
        dist = sqrt(deltaX ** 2 + deltaY ** 2)

        if dist != 0:
            # TODO care if spawned anyway it will not move
            dx = deltaX * self.speed / dist
            dy = deltaY * self.speed / dist
            self.body.x += dx
            self.body.y += dy

        # laser potential collision with ships
        explode = False
        for ship in self.battleground.ships:
            # colliding a ship
            # we pass through destroyed ships
            if ship.is_playable() and self.body.collide(ship.body):
                # Destroying other ships sparks joy in ship
                self.owner.agent.reward += REWARDS["kill"]
                self.owner.battleground.last_x_time_rewards.append((1, self.battleground.time))
                ship.hit(self)
                explode = True
        if explode or self.battleground.outside(self.body.x, self.body.y):
            self.explode()

    def explode(self):
        # print("prrrr", self.id)
        self.state = "destroyed"

    def __str__(self):
        return str(self.id)
        # return functools.reduce(lambda x : x + self. , self.)

    def __repr__(self):
        return str(self.id)



if __name__ == "__main__":
    Laser()