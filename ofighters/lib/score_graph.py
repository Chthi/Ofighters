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

from ofighters.lib.graph import Graph


class ScoreGraph(Graph):
    """Link to an object (a bot) with attributes .agent.scores
    where scores is a list of floats
    and display a graph of the values"""

    def __init__(self, master_window, bot):
        super().__init__()
        self.master = master_window
        self.bot = bot

        # making sure the given object has the correct attributes
        if not (hasattr(self.bot, "agent") and hasattr(self.bot.agent, "scores")):
            raise Exception("bot mut have attributes .agent.scores")


    def create(self):
        super().create()
        self.ax = self.figure.add_subplot(1,1,1)
        self.ax.set_xlabel('Episodes')
        self.ax.set_ylabel('Score')
        plt.tight_layout()
        self.len_scores = 1
        self.master.after(1000, self.animate)
        self.master.after(0, self.figure.show)


    def animate(self):
        """Must create a graph before calling animate"""
        if not self.created:
            warnings.warn("Must create a graph before calling animate")
        else:
            # only animate is the list has changed size
            if self.len_scores != len(self.bot.agent.scores):
                scores = self.bot.agent.scores
                self.len_scores = len(scores)
                self.ax.cla()
                # print("scores", scores)

                self.ax.plot(scores, label='Score', linewidth=1.5)
                window_len = 11
                half_window_len = (window_len-1)//2
                if len(scores) >= window_len :
                    # 'full', 'same', 'valid'
                    scores_10 = np.convolve(np.array(scores), np.ones((window_len,))/window_len, mode='valid')
                    # valid convolution loose border so we re-center
                    x = range(half_window_len, half_window_len + len(scores_10))
                    self.ax.plot(x, scores_10, label='Score on a {0} steps window'.format(window_len), color="green")
                self.ax.legend(loc='upper left')
                plt.pause(0.5)
            # called at each restart instead
            # self.master.after(500, self.animate)

