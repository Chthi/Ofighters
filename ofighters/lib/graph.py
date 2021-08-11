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


class Graph():

    def __init__(self):
        self.created = False

    def create(self, figure=None):
        self.figure = figure or plt.figure()
        self.cid = self.figure.canvas.mpl_connect('close_event', self.close)
        self.created = True
        debug("graph created")


    def animate(self):
        """Must create a graph before calling animate"""
        if not self.created:
            warnings.warn("Must create a graph before calling animate")
        else:
            debug("animate graph")
            plt.pause(3)


    def destroy(self):
        """Must create a graph before calling destroy"""
        if not self.created:
            raise Exception("Must create a graph before calling destroy")
        # will callback self.close
        print("closing graph")
        plt.close(self.figure)


    def close(self, event):
        self.created = False
        event.canvas.stop_event_loop()
        print("event closing graph")


if __name__ == "__main__":
    g = Graph()
    g.create()
    # plt.show()
    g.animate()
    if g.created:
        g.destroy()

