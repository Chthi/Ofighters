
import os
import sys
import math
import numpy as np
from itertools import count

import matplotlib
# While GTK isn't avail everywhere, we use TkAgg backend to generate png
if sys.platform != "win32" and os.getenv("DISPLAY") is None :
    matplotlib.use("Agg")
else :
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def cos_P_u(time, period_time, amplitude):
    return amplitude * ((math.cos((time / period_time) * 2 * math.pi) + 1) / 2)

# print(cos_P_u(0, 200, 1))
# print(cos_P_u(50, 200, 1))
# print(cos_P_u(100, 200, 1))
# print(cos_P_u(150, 200, 1))
# print(cos_P_u(200, 200, 1))

def reverse_cos_P_u(value, period_time, amplitude):
    return period_time * math.acos( ((2*value)/ amplitude) -1 ) / (2 * math.pi)

# print(reverse_cos_P_u(1, 200, 1))
# print(reverse_cos_P_u(0.5, 200, 1))
# print(reverse_cos_P_u(0, 200, 1))


class Epsilon_cos():

    def __init__(self, period):
        self.t = 0
        self.amplitude = 1
        self.period = period
        self.epsilon = cos_P_u(self.t, self.period, self.amplitude)

    def next(self):
        # t don't need to go to crazy high numbers
        self.t = (self.t + 1) % self.period
        self.epsilon = cos_P_u(self.t, self.period, self.amplitude)
        # print("next is ", self.epsilon, "t = ", self.t)
        return self.epsilon

    def get(self):
        return self.epsilon

    def set(self, value):
        if value > 1.0 or value < 0.0:
            raise Exception("Value must me in range [0,1]")
        self.epsilon = value
        self.t = reverse_cos_P_u(value, self.period, self.amplitude)
        print("set t to : ", self.t)


class Epsilon_decay():

    def __init__(self):
        # self.t = 0
        self.epsilon = 1
        # epsilon soft min. it can be smaller but not too much
        # if epsilon goes or is set to a smallest number it will stay like that
        self.epsilon_min = 0.01
        self.decay = 0.99990
        print("Time before decaying to 1% :", math.log(0.01) / math.log(self.decay), "iterations." )

    def next(self):
        # do not touch epsilon if below the threshold
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay
        # self.t += 1
        return self.epsilon

    def get(self):
        return self.epsilon

    def set(self, value):
        if value > 1.0 or value < 0.0:
            raise Exception("Value must me in range [0,1]")
        self.epsilon = value



if __name__ == "__main__":

    fig = plt.figure(figsize=(12, 5))
    ax_losses = fig.subplots(nrows=1, ncols=2)
    y = [cos_P_u(t, 200, 1) for t in range(0, 400)]

    epsilon = Epsilon_cos(110*400)
    # epsilon = Epsilon_decay()
    y = [epsilon.next() for t in range(0, 46000)]
    epsilon = Epsilon_cos(200)
    y = [epsilon.next() for t in range(0, 200)]

    print(y)
    # y = cos_P_u(400, 200, 1)
    # print("x.shape", len(x))
    # print("y.shape", len(y))

    x = range(0, len(y))
    ax_losses[0].plot(x, y)

    # plt.tight_layout()

    y = [reverse_cos_P_u(v/1000, 200, 1) for v in range(0, 1000)]
    x = [v/1000 for v in range(0, 1000)]
    ax_losses[1].plot(x, y)
    plt.show()

    # plt.pause(4)
