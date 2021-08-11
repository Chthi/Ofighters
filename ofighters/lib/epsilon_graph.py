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

from ofighters.lib.utils import debug
from ofighters.lib.graph import Graph


class EpsilonGraph(Graph):
    """Link to an object (a bot) with attributes .agent.epsilons
    where epsilons is a list of floats
    and display a graph of the values"""

    def __init__(self, master_window, bot):
        super().__init__()
        self.master = master_window
        self.bot = bot

        # making sure the given object has the correct attributes
        if not (hasattr(self.bot, "agent") and hasattr(self.bot.agent, "epsilons")):
            raise Exception("bot mut have attributes .agent.epsilons")


    def create(self):
        super().create()
        self.ax = self.figure.add_subplot(1,1,1)
        self.ax.set_xlabel('Episodes')
        self.ax.set_ylabel('Epsilon (exploration rate)')
        plt.tight_layout()
        self.len_epsilons = 0
        self.master.after(1000, self.animate)
        self.master.after(0, self.figure.show)


    def animate(self):
        """Must create a graph before calling animate"""
        if not self.created:
            warnings.warn("Must create a graph before calling animate")
        else:
            # only animate is the list has changed size
            if self.len_epsilons != len(self.bot.agent.epsilons):
                epsilons = self.bot.agent.epsilons
                self.len_epsilons = len(epsilons)
                self.ax.cla()
                # print("epsilons", epsilons)

                # x = range(len(epsilons))
                self.ax.plot(epsilons, label='Epsilon', linewidth=1.5) # color="magenta"
                self.ax.set_ylim(-0.1, 1.05)
                # if len(epsilons) > 10:
                    # 'full', 'same', 'valid'
                    # epsilons_10 = np.convolve(np.array(epsilons), np.ones((10,))/10, mode='same')
                    # self.ax.plot(epsilons_10, label='Epsilon on a 10 steps window', color="magenta")
                self.ax.legend(loc='upper left')
                plt.pause(0.5)
            # called at each restart instead
            # self.master.after(500, self.animate)
