
import os
import sys
import math

import numpy as np

import matplotlib
# While GTK isn't avail everywhere, we use TkAgg backend to generate png
if sys.platform != "win32" and os.getenv("DISPLAY") is None :
    matplotlib.use("Agg")
else :
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from form import Circle

def angle_tir(ship, point):
    pass

def angle_relatif(ship, other_ship):
    pass




def y_func(time, period_time, amplitude):
    return amplitude * ((math.cos((time / period_time) * 2 * math.pi) + 1) / 2)


fig = plt.figure() #figsize=(10, 5))
ax_losses = fig.subplots(nrows=1, ncols=1)
x = range(0, 400)
y = [y_func(t, 200, 1) for t in range(0, 400)]
print(y)
# y = y_func(400, 200, 1)
# print("x.shape", len(x))
# print("y.shape", len(y))

ax_losses.plot(x, y)

# plt.tight_layout()

plt.show()
# plt.pause(4)
