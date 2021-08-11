
import os
import sys

import numpy as np

import matplotlib
# While GTK isn't avail everywhere, we use TkAgg backend to generate png
if sys.platform != "win32" and os.getenv("DISPLAY") is None :
    matplotlib.use("Agg")
else :
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DEFAULT_WIDTH = 400
DEFAULT_HEIGHT = 400
ACTION_SIZE = 2

act_values = 9 * (np.random.random((ACTION_SIZE, 1)) - 0.5)
ptr_values = 9 * (np.random.random((DEFAULT_WIDTH, DEFAULT_HEIGHT)) - 0.5 )
# ptr_values = np.linspace(-8, 8, num=(DEFAULT_WIDTH * DEFAULT_HEIGHT)).reshape(DEFAULT_WIDTH, DEFAULT_HEIGHT)

iaction =  np.argmax(act_values)
ipointer = np.unravel_index(np.argmax(ptr_values, axis=None), ptr_values.shape)
print(ipointer)
print("ptr_values[ipointer]", ptr_values[ipointer])
print("act_values", act_values)

fig = plt.figure(figsize=(10, 5))
ax_losses = fig.subplots(nrows=1, ncols=2)
im = ax_losses[0].imshow(ptr_values, cmap='coolwarm', interpolation='bilinear', vmin=-8, vmax=8)
fig.colorbar(im)
ax_losses[0].plot(ipointer[0], ipointer[1], marker='X', markersize=10, markerfacecolor='r', markeredgecolor='black')
ax_losses[1].imshow(act_values, cmap='coolwarm', interpolation='nearest', vmin=-8, vmax=8)#, origin='lower', extent=[-1,1,-1,1], aspect=4)
ax_losses[1].axis('off')

plt.tight_layout()

plt.pause(4)
