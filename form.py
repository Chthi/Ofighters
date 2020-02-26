#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt

from skimage import draw

def inbounds(x, y, maxX, maxY):
    return (x >= 0) and (y >= 0) and (x < maxX) and (y < maxY)


def rotate(orientation, clockwise=True):
    """rotate a coordinate of pi/2
    orientation is between -pi and pi
    new orientation is between -pi and pi"""
    rotation = -math.pi/2 if clockwise else math.pi/2
    orientation = orientation + rotation
    if orientation > math.pi:
        orientation -= math.pi
    return orientation



class Forme:
    """Represent a form with a hitbox."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


    def distance(self, pos):
        """pos must have x and y attributs"""
        return math.sqrt((self.x - pos.x)**2 + (self.y - pos.y)**2)


    def angle_with(self, point, origin="down_left"):
        """Return a value in radians between -pi and pi
        If the 2 points are at the same position return None
        for the moment only work with origin = down_left
        where y increase by going up and x by going right"""
        dx = point.x - self.x
        dy = point.y - self.y
        res = 0
        if dx > 0:
            res = math.atan(dy/dx)
        elif dx < 0:
            if dy > 0:
                res = math.pi - math.atan(dy/abs(dx))
            elif dy < 0:
                res = - math.pi - math.atan(dy/abs(dx))
            else:
                res = math.pi
        else:
            if dy > 0:
                res = math.pi/2
            elif dy < 0:
                res = -math.pi/2
            else:
                res = None
        return res


    # def __rsub__(self, other):
    #     if isinstance(other, Forme):
    #         self.x -= other.x
    #         self.y -= other.y

    # def __radd__(self, other):
    #     if isinstance(other, Forme):
    #         self.x += other.x
    #         self.y += other.y
    #     else:
    #         self.x += other
    #         self.y += other

    # def __add__(self, other):
    #     if isinstance(other, Forme):
    #         return Forme(self.x + other.x, self.y + other.y)
    #     else:
    #         return Forme(self.x + other, self.y + other)

    # def __sub__(self, other):
    #     if isinstance(other, Forme):
    #         return Forme(self.x - other.x, self.y - other.y)
    #     else:
    #         return Forme(self.x - other, self.y - other)

    # def __mul__(self, other):
    #     if isinstance(other, Forme):
    #         return Forme(self.x * other.x, self.y * other.y)
    #     else:
    #         return Forme(self.x * other, self.y * other)

    # def __truediv__(self, other):
    #     if isinstance(other, Forme):
    #         return Forme(self.x / other.x, self.y / other.y)
    #     else:
    #         return Forme(self.x / other, self.y / other)

    # def __floordiv__(self, other):
    #     if isinstance(other, Forme):
    #         return Forme(self.x // other.x, self.y // other.y)
    #     else:
    #         return Forme(self.x // other, self.y // other)

    # def __repr__(self):
    #     return "({0}, {1})".format(self.x, self.y)

    # def __str__(self):
    #     return "({0}, {1})".format(self.x, self.y)

    # def copy(self):
    #     return Forme(self.x, self.y)




class Circle(Forme):
    
    def __init__(self, x, y, radius):
        super().__init__(x, y)
        self.radius = radius

    def collideCircle(self, circle):
        return self.distance(circle) <= self.radius + circle.radius

    def collide(self, body):
        if isinstance(body, Circle):
            collision = self.collideCircle(body)
        else:
            raise Exception("collision with other object than circle are not implemented")
        return collision


    def edge(self, xn, yn, distance=0):
        """
        find a position on the edge of the form such as it is on a line between its center and a given point (xn, yn)
        distance is the minimal distance between the edge of the form and the wanted position.
        when distance = 0 the position in on the edge

        If the point xn, yn is the center of the form return None.

        example of edges with distance=1 radius=2
                 0,3
           -2,2   0   2,2
                0 0 0
        -3,0  0 0 0 0 0  3,0
                0 0 0
           -2,-2  0   2,-2
                0,-3
        """

        # base + distance
        inR = self.radius + distance
        deltaX = xn - self.x
        deltaY = yn - self.y
        dist = math.sqrt(deltaX ** 2 + deltaY ** 2)

        if dist == 0:
            return None

        dx = deltaX * inR / dist
        dy = deltaY * inR / dist
        return (int(self.x + dx), int(self.y + dy))



    def span(self, grid_size_side):
        """
        PROTOTYPE
        return (t, r, d, l) respectively the number of cells hovered by the circle
        in a grid of given side size (excepting the cell where is the circle).
        we consider grids cells as squared shaped
        ex : grid_size_side = 2 and self.radius = 2
        @ : center
        o : pixels in same cell as center
        v : pixels ovelapping on other cells 
        , , . . , ,
        , , . . v ,
        . . , v @ o
        . . , , o .
        , , . . , ,
        , , . . , ,
        (t, r, d, l) = (1, 0, 0, 1)
        """
        # number of always overlapped cells
        depassement = self.radius // grid_size_side
        # radius we will compare with distances from the 4 borders
        residual_radius = self.radius % grid_size_side
        
        t = depassement + (1 if self.radius > self.y % grid_size_side else 0)
        r = depassement + (1 if self.radius > grid_size_side - (self.y % grid_size_side) else 0)
        d = depassement + (1 if self.radius > grid_size_side - (self.x % grid_size_side) else 0)
        l = depassement + (1 if self.radius > self.x % grid_size_side else 0)
        return (t, r, d, l)


    def binary_draw(self, grid):
        rr, cc = draw.circle(self.x, self.y, radius=self.radius, shape=grid.shape)
        grid[rr, cc] = 1
        return grid


    def binary_draw_old(self, grid, grid_size_side):
        """Depreciated"""
        maxX = grid.shape[0] - 1
        maxY = grid.shape[1] - 1
        # number of always overlapped cells
        depassement = self.radius // grid_size_side
        # radius we will compare with distances from the 4 borders
        residual_radius = self.radius % grid_size_side
        
        cellX = int(min(self.x // grid_size_side, maxX))
        cellY = int(min(self.y // grid_size_side, maxY))
       
        # draw centrer of the circle
        # . . . . .
        # . . . . .
        # . . o . .
        # . . . . .
        # . . . . .
        # print("cellX", cellX)
        # print("cellY", cellY)
        grid[cellX][cellY] = 1

        # draw 4 branches of the circle
        # . . o . .
        # . . o . .
        # o o X o o
        # . . o . .
        # . . o . .
        for n in range(1, depassement+1):
            # OPTIM maybe slow
            if inbounds(cellX+n, cellY, maxX, maxY) : grid[cellX+n][cellY] = 1
            if inbounds(cellX-n, cellY, maxX, maxY) : grid[cellX-n][cellY] = 1
            if inbounds(cellX, cellY+n, maxX, maxY) : grid[cellX][cellY+n] = 1
            if inbounds(cellX, cellY-n, maxX, maxY) : grid[cellX][cellY-n] = 1

        # fill quarters of the circle
        # . o X o .
        # o o X o o
        # X X X X X
        # o o X o o
        # . o X o .
        # for x in range(1, depassement+1):
        #     for y in range(1, depassement+1-x):
        #         # OPTIM maybe slow
        #         if inbounds(cellX+x, cellY+y, maxX, maxY) : grid[cellX+x][cellY+y] = 1
        #         if inbounds(cellX+x, cellY-y, maxX, maxY) : grid[cellX+x][cellY-y] = 1
        #         if inbounds(cellX-x, cellY+y, maxX, maxY) : grid[cellX-x][cellY+y] = 1
        #         if inbounds(cellX-x, cellY-y, maxX, maxY) : grid[cellX-x][cellY-y] = 1

        # add additional tips to branches depending on the center of the circle
        # . X X X .
        # X X X X X
        # X X X X X X    -> the original circle isinstance a bit centered down right
        # X X X X X
        # . X X X .
        # . . X . .
        # if self.radius > grid_size_side - (self.x % grid_size_side):
        #     grid[cellX+depassement+1][cellY] = 1
        # if self.radius > self.x % grid_size_side:
        #     grid[cellX-depassement-1][cellY] = 1
        # if self.radius > grid_size_side - (self.y % grid_size_side):
        #     grid[cellX][cellY+depassement+1] = 1
        # if self.radius > self.y % grid_size_side:
        #     grid[cellX][cellY-depassement-1] = 1

        return grid




import numpy as np

def test_forme():
    c1 = Circle(0, 0, 0.5)
    c2 = Circle(1, 1, 0.5)
    c3 = Circle(0, 1, 0.5)
    print(c1.distance(c2))
    print(c1.collide(c2))
    print(c1.distance(c3))
    print(c1.collide(c3))
    # see example of binary_draw docstring
    circle = Circle(4, 2, 2)
    grid = np.zeros((3,3))
    grid = circle.binary_draw(grid, 2)
    print("grid\n", grid.T)


def test_angle_with():
    # using down left origin
    origin = Circle(0, 0, 1)
    cNone = Circle(0, 0, 1)
    c0 = Circle(1, 0, 1)
    c1_4 = Circle(1, 1, 1)
    c1_2 = Circle(0, 1, 1)
    c3_4 = Circle(-1, 1, 1)
    cpi = Circle(-1, 0, 1)
    c_3_4 = Circle(-1, -1, 1)
    c_1_2 = Circle(0, -1, 1)
    c_1_4 = Circle(1, -1, 1)
    # None
    print("{} : points superposed".format(origin.angle_with(cNone, origin="down_left")))
    # 0 pi
    print("{} pi".format(origin.angle_with(c0, origin="down_left") / math.pi))
    # 1/4 pi
    print("{} pi".format(origin.angle_with(c1_4, origin="down_left") / math.pi))
    # 1/2 pi
    print("{} pi".format(origin.angle_with(c1_2, origin="down_left") / math.pi))
    # 3/4 pi
    print("{} pi".format(origin.angle_with(c3_4, origin="down_left") / math.pi))
    # pi
    print("{} pi".format(origin.angle_with(cpi, origin="down_left") / math.pi))
    # -3/4 pi
    print("{} pi".format(origin.angle_with(c_3_4, origin="down_left") / math.pi))
    # -1/2 pi
    print("{} pi".format(origin.angle_with(c_1_2, origin="down_left") / math.pi))
    # -1/4 pi
    print("{} pi".format(origin.angle_with(c_1_4, origin="down_left") / math.pi))


def test_rotate():
    print("{} pi".format(rotate(0) / math.pi))
    print("{} pi".format(rotate(math.pi/2) / math.pi))
    print("{} pi".format(rotate(math.pi) / math.pi))
    print("{} pi".format(rotate(-math.pi/2) / math.pi))

    print("{} pi".format(rotate(0, False) / math.pi))
    print("{} pi".format(rotate(math.pi/2, False) / math.pi))
    print("{} pi".format(rotate(math.pi, False) / math.pi))
    print("{} pi".format(rotate(-math.pi/2, False) / math.pi))


if __name__ == '__main__':
    # test_forme()
    # test_angle_with()
    test_rotate()