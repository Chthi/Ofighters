#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import warnings

import numpy as np

import matplotlib
# While GTK isn't avail everywhere, we use TkAgg backend to generate png
if sys.platform != "win32" and os.getenv("DISPLAY") is None :
    matplotlib.use("Agg")
else :
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from graph import Graph


class LossGraph(Graph):
    """Link to an object (a bot) with attributes .agent.losses
    where losses is a list of floats
    and display a graph of the values"""

    def __init__(self, master_window, bot):
        super().__init__()
        self.master = master_window
        self.bot = bot

        # making sure the given object has the correct attributes
        if not (hasattr(self.bot, "agent") and hasattr(self.bot.agent, "losses")):
            raise Exception("bot mut have attributes .agent.losses")


    def create(self):
        super().create()
        self.ax = self.figure.add_subplot(1,1,1)
        self.ax.set_xlabel('Replays')
        self.ax.set_ylabel('Loss')
        plt.tight_layout()
        self.len_losses = 1
        self.master.after(1000, self.animate)
        self.master.after(0, self.figure.show)


    def animate(self):
        """Must create a graph before calling animate"""
        # we stop the looping and drawing process if the window have been closed
        # or if create have not been called
        if not self.created:
            # warnings.warn("Must create a graph before calling animate")
            return

        # only animate is the list has changed size
        if self.len_losses != len(self.bot.agent.losses):
            self.draw()
            plt.pause(0.5)
        self.master.after(500, self.animate)


    def draw(self):
        losses = self.bot.agent.losses
        self.len_losses = len(losses)
        self.ax.cla()
        print("losses", losses)

        self.ax.plot(losses, label='Loss', linewidth=1.5)
        window_len = 11
        half_window_len = (window_len-1)//2
        if len(losses) >= window_len :
            # 'full', 'same', 'valid'
            losses_10 = np.convolve(np.array(losses), np.ones((window_len,))/window_len, mode='valid')
            # valid convolution loose border so we re-center
            x = range(half_window_len, half_window_len + len(losses_10))
            self.ax.plot(x, losses_10, label='Loss on a {0} steps window'.format(window_len))
        self.ax.legend(loc='upper left')

        # self.line_losses.set_data(range(len(losses)), losses) # set plot data
        # self.ax.lines[0].set_data(range(len(losses)), losses) # set plot data
        # self.ax.relim()                  # recompute the data limits
        # self.ax.autoscale_view()         # automatic axis scaling
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()   # update the plot and take care of window events (like resizing etc.)
        # sleep(0.2)               # wait for next loop iteration
