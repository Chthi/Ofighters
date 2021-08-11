#!/usr/bin/python
# -*- coding: utf-8 -*-


class Couple():
    """Represent a pair of values.
    Can be used as coordinates."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __rsub__(self, other):
        if isinstance(other, Couple):
            self.x -= other.x
            self.y -= other.y

    def __radd__(self, other):
        if isinstance(other, Couple):
            self.x += other.x
            self.y += other.y
        else:
            self.x += other
            self.y += other

    def __add__(self, other):
        if isinstance(other, Couple):
            return Couple(self.x + other.x, self.y + other.y)
        else:
            return Couple(self.x + other, self.y + other)

    def __sub__(self, other):
        if isinstance(other, Couple):
            return Couple(self.x - other.x, self.y - other.y)
        else:
            return Couple(self.x - other, self.y - other)

    def __mul__(self, other):
        if isinstance(other, Couple):
            return Couple(self.x * other.x, self.y * other.y)
        else:
            return Couple(self.x * other, self.y * other)

    def __truediv__(self, other):
        if isinstance(other, Couple):
            return Couple(self.x / other.x, self.y / other.y)
        else:
            return Couple(self.x / other, self.y / other)

    def __floordiv__(self, other):
        if isinstance(other, Couple):
            return Couple(self.x // other.x, self.y // other.y)
        else:
            return Couple(self.x // other, self.y // other)

    def __repr__(self):
        return "Couple({0}, {1})".format(self.x, self.y)

    def __str__(self):
        return "Couple({0}, {1})".format(self.x, self.y)

    def copy(self):
        return Couple(self.x, self.y)

    def toList(self):
        return [self.x, self.y]

    def toTuple(self):
        return (self.x, self.y)

    def toArray(self):
        try:
            import numpy
        except ImportError:
            raise NotImplementedError("You need the numpy package to use this method.")
        else:
            return numpy.array([self.x, self.y])


class Point(Couple):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "Point({0}, {1})".format(self.x, self.y)

    def __str__(self):
        return "Point({0}, {1})".format(self.x, self.y)



def test_couple():
    c1 = Couple(1, 1)
    c2 = Couple(2, 2)
    print(" ---- {} ---- ".format("c1 + c2"))
    print(c1 + c2)
    print(" ---- {} ---- ".format("c1 - c2"))
    print(c1 - c2)
    print(" ---- {} ---- ".format("c1 * c2"))
    print(c1 * c2)
    print(" ---- {} ---- ".format("c1 to list"))
    print(c1.toList())
    print(" ---- {} ---- ".format("c1 to tuple"))
    print(c1.toTuple())
    print(" ---- {} ---- ".format("c2 to numpy array"))
    print(c2.toArray())
    print(" ---- {} ---- ".format("couple of couple"))
    cc = Couple(c1, c2)
    print(cc)
    print(" ---- {} ---- ".format("cc + cc"))
    print(cc + cc)
    print(" ---- {} ---- ".format("cc to list"))
    print(cc.toList())

def test_point():
    p1 = Point(1, 1)
    p2 = Point(2, 2)
    print(" ---- {} ---- ".format("p1 + p2"))
    print(p1 + p2)
    print(" ---- {} ---- ".format("p1 - p2"))
    print(p1 - p2)
    print(" ---- {} ---- ".format("p1 * p2"))
    print(p1 * p2)
    print(" ---- {} ---- ".format("p1 to list"))
    print(p1.toList())
    print(" ---- {} ---- ".format("p1 to tuple"))
    print(p1.toTuple())
    print(" ---- {} ---- ".format("p2 to numpy array"))
    print(p2.toArray())


if __name__ == '__main__':
    test_couple()
    # test_point()
