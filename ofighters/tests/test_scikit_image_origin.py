#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import warnings

import matplotlib
# While GTK isn't avail everywhere, we use TkAgg backend to generate png
if sys.platform != "win32" and os.getenv("DISPLAY") is None :
    matplotlib.use("Agg")
else :
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
from skimage import draw



map = np.zeros((100, 100))
rr, cc = draw.disk((10, 20), radius=10, shape=map.shape)
map[rr, cc] = 1
print(str(map))

plt.imshow(map, cmap='binary', interpolation='none', origin='upper')
# plt.imshow(map, cmap='binary', interpolation='none', origin='lower')


plt.pause(10)