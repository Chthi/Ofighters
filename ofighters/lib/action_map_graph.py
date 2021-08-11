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
from ofighters.lib.utils import debug


class ActionMapGraph(Graph):
    """Link to an object (a bot) with attributes .agent.trainer.ptr_values and .agent.trainer.act_values
    where ptr_values is a 2D vector of floats
    and act_values a list of floats
    and display a graph of the values"""

    def __init__(self, master_window, bot):
        super().__init__()
        self.master = master_window
        self.bot = bot

        # making sure the given object has the correct attributes
        if not (hasattr(self.bot, "agent") and hasattr(self.bot.agent, "trainer") and
                hasattr(self.bot.agent.trainer, "ptr_values") and hasattr(self.bot.agent.trainer, "act_values")):
            raise Exception("bot mut have attributes .agent.trainer.ptr_values and .agent.trainer.act_values")
        # TODO test input obs var


    def create(self):
        # super().create(figure=plt.figure(figsize=(10, 5)))
        super().create(figure=plt.figure(figsize=(10, 10)))
        # self.ax = self.figure.subplots(nrows=1, ncols=2)
        self.ax = self.figure.subplots(nrows=2, ncols=2)
        # plt.tight_layout()
        self.colorbar_ptr = None
        self.colorbar_act = None
        self.master.after(1000, self.animate)
        self.master.after(0, self.figure.show)


    def animate(self):
        """Must create a graph before calling animate"""
        # we stop the looping and drawing process if the window have been closed
        # or if create have not been called
        if not self.created:
            # warnings.warn("Must create a graph before calling animate")
            return

        if not (self.bot.agent.trainer.ptr_values is not None and self.bot.agent.trainer.act_values is not None):
            # debug("bot ptr_values and act_values must not be None : ignoring this animation")
            # warnings.warn("bot ptr_values and act_values must not be None : ignoring this animation")
            pass
        else:
            print("--- ACTUALISATION ---")
            self.draw_input_map()
            self.draw_action_map()
            plt.pause(0.9)
        self.master.after(100, self.animate)


    def draw_input_map(self):
        obs = self.bot.agent.previous_obs

        self.ax[0][0].cla()
        self.ax[0][1].cla()
        self.img_ships = self.ax[0][0].imshow(obs.ship_map, cmap='binary', interpolation='none', origin='upper') # vmin=-8, vmax=8
        # TODO pointer on ship position
        # self.ax[0][0].plot(ipointer[0], ipointer[1], marker='x', markersize=10, markerfacecolor='r', markeredgecolor='black')
        self.img_lasers = self.ax[0][1].imshow(obs.laser_map, cmap='binary', interpolation='none', origin='upper')#extent=[-1,1,-1,1], aspect=4)
        # self.ax[0].axis('off')
        self.ax[0][1].axis('off')


    def draw_action_map(self):
        # TODO only animate is the list has changed size ?
        ptr_values = self.bot.agent.trainer.ptr_values
        act_values = self.bot.agent.trainer.act_values
        iaction = np.argmax(act_values)
        act_values = np.expand_dims(act_values, axis=1)
        ipointer = np.unravel_index(np.argmax(ptr_values, axis=None), ptr_values.shape[0:2], order='F')

        # print("prediction", act_values)
        # print("ptr_values", ptr_values.shape)
        actions_list = ["shoot", "thrust", "turn"]
        print("action", actions_list[iaction])
        # print("pointer", ipointer)

        self.ax[1][0].cla()
        self.ax[1][1].cla()
        self.img_ptr = self.ax[1][0].imshow(ptr_values, cmap='coolwarm', interpolation='none', origin='upper') # vmin=-8, vmax=8
        self.ax[1][0].plot(ipointer[0], ipointer[1], marker='x', markersize=10, markerfacecolor='r', markeredgecolor='black')
        self.img_act = self.ax[1][1].imshow(act_values, cmap='OrRd', interpolation='none', origin='upper')#extent=[-1,1,-1,1], aspect=4)
        # self.ax[0].axis('off')
        self.ax[1][1].axis('off')

        if not self.colorbar_ptr:
            self.colorbar_ptr = self.figure.colorbar(self.img_ptr, ax=self.ax[1][0], extend='both')
            self.colorbar_act = self.figure.colorbar(self.img_act, ax=self.ax[1][1], extend='both')
        else:
            self.img_ptr.set_clim(vmin=ptr_values.min(),vmax=ptr_values.max())
            self.img_act.set_clim(vmin=act_values.min(),vmax=act_values.max())
            # self.colorbar_ptr.set_ticks(np.linspace(0., 2., num=11, endpoint=True))
            # self.colorbar_ptr.draw_all()

        # self.ax.legend(loc='upper left')