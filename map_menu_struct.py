#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os

from collections import namedtuple, deque

from time import sleep, time
from random import random

import tkinter as tk
from tkinter import ttk
from tkinter import *

from couple import Couple
from popups import Alert

from thread_manager import spawnthread

# minimum sleep time between frames to garranty to not freeze the machine
MIN_SLEEP = 0.001 # 1 ms



def counter():
    """generate all the number from 0 up to infinite"""
    n = 0
    while True:
        n += 1
        yield n


class MapMenuStruct():
    """Standard structure for the graphical component.
    Any object inheriting this one will be a graphical interface.
    Features can be added, overwritten and command should be changed."""

    def __init__(self, largeur=800, hauteur=800):
        # all the objects, parts and widgets of the interface
        self.ihm = dict()

        # attibutes
        self.dim = Couple(largeur, hauteur)

        # actualizing variables
        self.temps = 0
        self.episode = 1
        self.fps = 1
        self.sleep_time = 1
        
        # modes on/off
        self.quitter = False
        self.training_mode = False
        self.is_learning = True
        self.continuous_training = False


    def launch(self):
        # main window
        self.master = tk.Tk()

        # fill the main window
        self.create_main_window()

        # other windows
        self.threads = []
        self.threads.append(spawnthread(self.run))
        
        # run the main thread/loop
        self.master.mainloop()


    def create_main_window(self):
        """fill the main window of widgets"""
        self.last_fps_refresh_time = time()
        self.ihm["fps"] = tk.Label(self.master, text="FPS "+str(self.fps), font="Arial 8", anchor='ne')
        # self.ihm["fps"].pack()
        self.ihm["fps"].grid(row=0, column=0, sticky=SW)

        self.ihm["episode"] = tk.Label(self.master, text="Episode : "+str(self.episode), font="Arial 8", anchor='ne')
        # self.ihm["episode"].pack()
        self.ihm["episode"].grid(row=0, column=1, sticky=S)

        self.ihm["temps"] = tk.Label(self.master, text="Temps : "+str(self.temps), font="Arial 8", anchor='ne')
        # self.ihm["temps"].pack()
        self.ihm["temps"].grid(row=0, column=2, sticky=SE)

        self.ihm["carte"] = tk.Canvas(self.master,width=self.dim.x,height=self.dim.y, borderwidth=0, background="black")
        # self.ihm["carte"].pack()
        self.ihm["carte"].grid(row=1, columnspan=3, column=0)
        self.place_menu()


    def place_menu(self):
        self.latRight = tk.Frame()
        # self.latRight.pack()
        self.latRight.grid(row=0, rowspan=2, column=3, sticky=NSEW)
        self.latRightRowCount = counter()
        # self.place_speed()
        self.place_log_speed()
        self.place_transfer_player()
        # self.place_supervised()
        self.place_unsupervised()
        self.place_restart()
        self.place_quit()


    def place_speed(self):
        self.ihm["label_speed"] = tk.Label(self.latRight, 
            text="Speed", font="Ubuntu 14")
        self.ihm["label_speed"].grid(row=next(self.latRightRowCount), column=0, pady=(35,0))

        self.ihm["vitesse"] = tk.Scale(
            self.latRight, orient="horizontal", 
            from_=1, to=100, resolution=1, 
            tickinterval=100, length=100
            )
        self.ihm["vitesse"].set(5)
        # self.ihm["vitesse"].pack()
        self.ihm["vitesse"].grid(row=next(self.latRightRowCount), column=0)


    def place_log_speed(self):
        self.ihm["label_speed"] = tk.Label(self.latRight,
            text="Speed", font="Ubuntu 14")
        self.ihm["label_speed"].grid(row=next(self.latRightRowCount), column=0, pady=(20,0))

        self.ihm["label_log_speed"] = tk.Label(self.latRight, text="")
        self.ihm["label_log_speed"].grid(row=next(self.latRightRowCount), column=0)#, pady=(35,5))

        def actualise_label(value):
            self.ihm["label_log_speed"].config(text=f"{10**float(value):.1f}")

        self.ihm["vitesse"] = tk.Scale(
            self.latRight, orient="horizontal",
            from_=-0.3, to=3, resolution=0.1,
            tickinterval=0, length=100, showvalue=0,
            command=actualise_label
            )

        self.ihm["vitesse"].set(0.7)
        # self.ihm["vitesse"].pack()
        self.ihm["vitesse"].grid(row=next(self.latRightRowCount), column=0)


    def place_transfer_player(self):
        self.ihm["string_transfer_player"] = tk.StringVar()
        self.ihm["check_transfer_player"] = tk.Checkbutton(
            self.latRight, variable=self.ihm["string_transfer_player"], 
            text='Transfert the player', 
            onvalue='yes', offvalue='no', anchor='sw'
        )
        # self.ihm["check_transfer_player"].pack(side=tk.LEFT)
        self.ihm["check_transfer_player"].grid(row=next(self.latRightRowCount), column=0)
        self.ihm["check_transfer_player"].select()


    def place_supervised(self):
        # define elements
        def place_title():
            self.ihm["label_supervised"] = tk.Label(self.latRight,
                text="Supervised learning", font="Ubuntu 14")
            self.ihm["label_supervised"].grid(row=next(self.latRightRowCount), column=0, pady=(20,0))

        def place_button_recording():
            # check button to record actions taken by the player
            self.ihm["string_recording"] = tk.StringVar()
            self.ihm["check_recording"] = tk.Checkbutton(
                self.latRight, variable=self.ihm["string_recording"],
                text='Record',
                onvalue='yes', offvalue='no', anchor='sw'
            )
            # self.ihm["check_recording"].pack(side=tk.LEFT)
            self.ihm["check_recording"].grid(row=next(self.latRightRowCount), column=0)
            self.ihm["check_recording"].deselect()

        def place_button_train():
            # button to use recorded actions to train the network
            self.ihm["train"] = tk.Button(self.latRight, text="Train")
            # self.ihm["train"].pack()
            self.ihm["train"].grid(row=next(self.latRightRowCount), column=0)

        def place_progressbar():
            self.ihm["progress"] = ttk.Progressbar(self.latRight, orient="horizontal",
                                    length=200, mode="determinate")
            # we save the row on the grid where the loading bar must be placed
            # TODO automatize that for everything
            self.ihm["grid_raw_progress"] = next(self.latRightRowCount)
            self.ihm["progress"]["maximum"] = 100
            # self.ihm["progress"].grid(row=self.ihm["grid_raw_progress"], column=0)
            self.ihm["progress"]["value"] = 0

        def place_button_saveIA():
            # button to save the curent working ia
            # TODO what if multiple are running ?
            self.ihm["save_ia"] = tk.Button(self.latRight, text="Save this IA")
            # self.ihm["save_ia"].pack()
            self.ihm["save_ia"].grid(row=next(self.latRightRowCount), column=0)

        # place elements
        place_title()
        place_button_recording()
        place_button_train()
        place_progressbar()
        place_button_saveIA()


    def place_unsupervised(self):
        # define elements
        def place_title():
            self.ihm["label_unsupervised"] = tk.Label(self.latRight,
                text="Unsupervised learning", font="Ubuntu 14")
            self.ihm["label_unsupervised"].grid(row=next(self.latRightRowCount), column=0, pady=(20,0))

        def place_button_training_mode():
            # button to switch to training mode
            # no graphical visualisation, maximum speeds
            self.ihm["string_training_mode"] = tk.StringVar()
            self.ihm["check_training_mode"] = tk.Checkbutton(
                self.latRight, variable=self.ihm["string_training_mode"], text='Training mode',
                onvalue='yes', offvalue='no', anchor='sw', command=self.swap_training_mode
            )
            # self.ihm["check_training_mode"].pack(side=tk.LEFT)
            self.ihm["check_training_mode"].grid(row=next(self.latRightRowCount), column=0)
            self.ihm["check_training_mode"].deselect()

        def place_button_is_learning():
            # button to switch to is_learning mode
            # no regular history training
            self.ihm["string_is_learning"] = tk.StringVar()
            self.ihm["check_is_learning"] = tk.Checkbutton(
                self.latRight, variable=self.ihm["string_is_learning"], text='Learning',
                onvalue='yes', offvalue='no', anchor='sw', command=self.swap_is_learning
            )
            # self.ihm["check_is_learning"].pack(side=tk.LEFT)
            self.ihm["check_is_learning"].grid(row=next(self.latRightRowCount), column=0)
            self.ihm["check_is_learning"].select()


        def place_button_continuous_training():
            # allows to restart the simulation every few times
            self.ihm["string_continuous_training"] = tk.StringVar()
            self.ihm["check_continuous_training"] = tk.Checkbutton(
                self.latRight, variable=self.ihm["string_continuous_training"], text='Continuous training',
                onvalue='yes', offvalue='no', anchor='sw', command=self.swap_continuous_training
            )
            # self.ihm["check_continuous_training"].pack(side=tk.LEFT)
            self.ihm["check_continuous_training"].grid(row=next(self.latRightRowCount), column=0)
            self.ihm["check_continuous_training"].deselect()

        def place_exploration_rate():
            self.ihm["label_exploration_rate"] = tk.Label(self.latRight,
                text="Exploration", font="Ubuntu 12")
            self.ihm["label_exploration_rate"].grid(row=next(self.latRightRowCount), column=0)#, pady=(35,5))

            # we set a value linked to the scale so we can set the value without triggering the callback command
            exploration_value = tk.DoubleVar(self.master, name="exploration_value")
            self.ihm["exploration"] = tk.Scale(
                self.latRight, orient="horizontal",
                from_=0, to=1, resolution=0.01,
                tickinterval=1, length=100,
                variable=exploration_value
                )
            self.ihm["exploration"].set(1)
            # print(self.ihm["exploration"].getvar(name="exploration_value"))
            # self.ihm["exploration"].pack()
            self.ihm["exploration"].grid(row=next(self.latRightRowCount), column=0)

        # place elements
        place_title()
        place_button_training_mode()
        place_button_is_learning()
        place_button_continuous_training()
        # self.place_switch_session()
        place_exploration_rate()


    def place_switch_session(self):
        def ask_session_name():
            Alert("New session", "Create", callback=lambda x: print(x))
        self.ihm["switch_session"] = tk.Button(self.latRight, text="New session", command=ask_session_name)
        # self.ihm["switch_session"].pack()
        self.ihm["switch_session"].grid(row=next(self.latRightRowCount), column=0)


    def place_quit(self):
        self.ihm["quitter"] = tk.Button(self.latRight, text="Stopper la simulation", command=self.quit)
        # self.ihm["quitter"].pack()
        self.ihm["quitter"].grid(row=next(self.latRightRowCount), column=0, pady=(20,10)) # 410


    def hide_map(self):
        self.training_mode = True
        self.sleep_time = MIN_SLEEP
        # remove display interface
        # self.ihm["carte"].pack_forget()
        self.ihm["carte"].grid_forget()

    def expand_map(self):
        self.training_mode = False
        # self.sleep_time = 1 / self.ihm["vitesse"].get()
        self.sleep_time = 1 / (10 ** self.ihm["vitesse"].get())
        # put back display interface
        # self.ihm["carte"].pack()
        self.ihm["carte"].grid(row=1, columnspan=3, column=0)


    def swap_training_mode(self):
        if self.ihm["string_training_mode"].get() == "yes":
            self.hide_map()
        elif self.ihm["string_training_mode"].get() == "no":
            self.expand_map()


    def swap_is_learning(self):
        if self.ihm["string_is_learning"].get() == "yes":
            self.is_learning = True
        elif self.ihm["string_is_learning"].get() == "no":
            self.is_learning = False

    def swap_continuous_training(self):
        if self.ihm["string_continuous_training"].get() == "yes":
            self.continuous_training = True

        elif self.ihm["string_continuous_training"].get() == "no":
            self.continuous_training = False


    def place_restart(self):
        def next_episode():
            self.episode += 1
            self.temps = 0
        self.ihm["restart"] = tk.Button(self.latRight, text="Redémarrer", command=next_episode)
        # self.ihm["restart"].pack()
        self.ihm["restart"].grid(row=next(self.latRightRowCount), column=0, pady=(20,0))


    def quit(self):
        self.quitter = True


    def run(self):
        while not self.quitter:
            self.temps += 1
            self.ihm["temps"]["text"] = "Temps : "+str(self.temps)

            self.ihm["episode"]["text"] = "Episode : "+str(self.episode)

            # simulate computing
            sleep(random() / 20)

            if not self.training_mode:
                self.sleep_time = 1 / (10 ** self.ihm["vitesse"].get())

            sleep(self.sleep_time)

            self.ihm["fps"]["text"] = "FPS "+str(self.fps)

        self.master.destroy()




if __name__ == '__main__':
    mms = MapMenuStruct(400, 400)
    mms.launch()