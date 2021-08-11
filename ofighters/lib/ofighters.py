#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from collections import namedtuple, deque
from copy import copy, deepcopy

from time import sleep, time
import tkinter as tk
import numpy as np
import pickle
import json
import glob
import threading
from itertools import count
from statistics import mean, stdev

from ofighters.lib.record import OfighterRecord
from ofighters.lib.thread_manager import spawnthread
from ofighters.lib.couple import Couple
from ofighters.lib.form import Circle
from ofighters.lib.player import Player
from ofighters.lib.battleground import Battleground
from ofighters.lib.laser import Laser
from ofighters.lib.action import Action
from ofighters.lib.utils import now, debug
from ofighters.lib.observation import DEFAULT_WIDTH, DEFAULT_HEIGHT
from ofighters.lib.ship_image import ShipImage
from ofighters.lib.epsilon_graph import EpsilonGraph
from ofighters.lib.score_graph import ScoreGraph
from ofighters.lib.action_map_graph import ActionMapGraph
from ofighters.lib.loss_graph import LossGraph

# from brAIn import NETWORKS_FOLDER, BrAIn, SimpleModel
NETWORKS_FOLDER = "networks"

from ofighters.lib.map_menu_struct import MapMenuStruct
# from fps_manager import fps, fps_manager

import matplotlib
# While GTK isn't avail everywhere, we use TkAgg backend to generate png
if sys.platform != "win32" and os.getenv("DISPLAY") is None :
    matplotlib.use("Agg")
else :
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# from memory_usage import measure_size

# the potential player in included in random bots
SHIPS_MAP = {"idle" : 6, "QlearnIA" : 1}
# SHIPS_MAP = {"random" : 6, "QlearnIA" : 1}
# SHIPS_MAP = {"random" : 6}

MAP_PRECISION = 4

MAX_TIME = 200
ANTICIPATION = 100

RECORD_FOLDER = "ofighter_records"

DISPLAY_LOSSES = True
DISPLAY_SCORES = True
DISPLAY_EPSILONS = True
DISPLAY_ACTIONS_MAP = True


# coordinates conventions
#   | 0 | 1 | 2 | 3 | 4 -> x
# 0 |
# 1 |          pi/2
# 2 |           |
# 3 |   (-)pi _   _ 0
# 4 |           |
# y            -pi/2



class Ofighters(MapMenuStruct):
    """Main program and controller"""
    # fps_manager = fps_manager()

    def __init__(self, largeur=DEFAULT_WIDTH, hauteur=DEFAULT_HEIGHT):
        # we call the basic graphic interface for the game
        super().__init__(largeur, hauteur)

        self.session_name = ""
        # if we record the player ingame
        self.recording = False
        # self.training_mode = False
        self.record = None
        # when true the game will be restarted and passed
        # to the next episode as soon as possible
        # and then passed to false
        self.need_restart = False
        # contains all the objects we will need to delete
        self.todelete = {}

        self.frames = deque(maxlen=100)
        self.start = time()

        plt.style.use('fivethirtyeight')

        self.battleground = Battleground(ships=SHIPS_MAP, largeur=self.dim.x, hauteur=self.dim.y)

        # optional if at least one QlearnIA bot is present
        self._super_bot_one = None
        for ship in self.battleground.ships:
            # print("ship.agent.behavior", ship.agent.behavior)
            if ship.agent.behavior == "QlearnIA":
                # reference. read only. should not be modified.
                self._super_bot_one = ship
                break

        self.switch_session("default")

        Images = namedtuple("Images", ['ships', 'lasers', 'targets', ])
        # ships_images_list = len(self.battleground.ships) * ShipImage(self.ihm, self.battleground)
        ships_images_list = [ShipImage(self.ihm, self.battleground) for _ in self.battleground.ships]
        targets_images_list = len(self.battleground.ships) * [None]
        self.images = Images(ships_images_list, [], targets_images_list, )

        # images contains tkinter graphical objects
        # lasers contains (index, object) indexes of ships images
        # and corresponding object in the battleground
        # lasers contains indexes of ships images
        self.todelete = {"images" : [], "lasers" : [], "ships" : [], "targets" : []}

        self.threads = {}

        # main window
        self.master = tk.Tk()

        # fill the main window
        self.create_main_window()

        # self.create_widgets()
        self.transfer_player_ship()
        self.player_ship = True

        # linking Ofighters functions to the main graphical structure
        self.link_functionnalities()

        # def hide():
        #     self.ihm["check_training_mode"].select()
        #     self.swap_training_mode()
        # self.master.after(1000, hide)

        print("Main thread : ", threading.current_thread().ident)
        # other windows
        self.run()
        # self.master.after(0, self.run)
        # self.threads["run"] = spawnthread(self.run)

        # losses graphic window
        if DISPLAY_LOSSES:
            self.loss_graph = LossGraph(self.master, self._super_bot_one)
            self.loss_graph.create()
            # self.init_losses_graph()

        # scores graphic window
        if DISPLAY_SCORES:
            self.score_graph = ScoreGraph(self.master, self._super_bot_one)
            self.score_graph.create()
            # self.init_scores_graph()

        # epsilons (exploration rate) graphic window
        if DISPLAY_EPSILONS:
            self.epsilon_graph = EpsilonGraph(self.master, self._super_bot_one)
            self.epsilon_graph.create()
            # self.init_epsilons_graph()

        # actions and pointer map graphic window
        if DISPLAY_ACTIONS_MAP:
            self.action_map_graph = ActionMapGraph(self.master, self._super_bot_one)
            self.action_map_graph.create()
            # self.init_actions_map()

        # run the main thread/loop
        print("mainloop")
        self.master.mainloop()


    def init_losses_graph(self):
        # plt.ion()
        self.fig_losses = plt.figure()
        self.ax_losses = self.fig_losses.add_subplot(1,1,1)
        # self.ax_losses = plt.subplot(111)
        self.ax_losses.set_xlabel('Replays')
        self.ax_losses.set_ylabel('Loss')
        self.len_losses = 1
        # self.line_losses, = self.ax_losses.plot([0], [0])
        plt.tight_layout()
        self.master.after(1000, self.animate_loss)
        # self.master.after(0, plt.show)
        self.master.after(0, self.fig_losses.show)
        print("showed losses")

    def init_scores_graph(self):
        self.fig_scores = plt.figure()
        self.ax_scores = self.fig_scores.add_subplot(1,1,1)
        self.ax_scores.set_xlabel('Episodes')
        self.ax_scores.set_ylabel('Score')
        self.len_scores = 1
        plt.tight_layout()
        self.master.after(1000, self.animate_score)
        self.master.after(0, self.fig_scores.show)
        print("showed scores")

    def init_epsilons_graph(self):
        self.fig_epsilons = plt.figure()
        self.ax_epsilons = self.fig_epsilons.add_subplot(1,1,1)
        self.ax_epsilons.set_xlabel('Episodes')
        self.ax_epsilons.set_ylabel('Epsilon (exploration rate)')
        self.len_epsilons = 1
        plt.tight_layout()
        self.master.after(1000, self.animate_epsilon)
        self.master.after(0, self.fig_epsilons.show)
        print("showed epsilons")

    def init_actions_map(self):
        self.fig_actions_map = plt.figure(figsize=(10, 5))
        self.ax_actions_map = self.fig_actions_map.subplots(nrows=1, ncols=2)
        # plt.tight_layout()
        self.colorbar_ptr = None
        self.colorbar_act = None
        self.master.after(1000, self.animate_actions_map)
        self.master.after(0, self.fig_actions_map.show)
        print("showed actions_map")

    def restart(self):
        print("restart")
        if self.recording:
            self.save_records(os.path.join(RECORD_FOLDER, "ofighter_record_" + now()))
            # TODO check is nb ships diminue quand ils meurent
            self.record = OfighterRecord(self.battleground.absolute_state, len(self.battleground.ships), Battleground)
        networks = self.clear_battleground()
        # lobotomising the ships brains so they don't remember
        # the atrocities they have just seen
        for network in networks:
            network.clear_history()
        # regenerate a battleground
        print("battleground restart")
        self.battleground.restart()
        # self.battleground = Battleground(ship_number=SHIPS_NUMBER, hauteur=self.dim.x, largeur=self.dim.y)
        # we put back the player ship on the field if required
        # not used anymore as the ships objects are not deleted and the player stays in the ship until destroyed
        # if self.ihm["string_transfer_player"].get() == "yes" and self.ihm["string_training_mode"].get() == "no":
        #     self.transfer_player_ship()
        if DISPLAY_SCORES:
            if self.ihm["string_training_mode"].get() == "no" and self.score_graph.created:
                self.score_graph.animate()
                # self.animate_score()
        if DISPLAY_EPSILONS:
            if self.ihm["string_training_mode"].get() == "no" and self.epsilon_graph.created:
                self.epsilon_graph.animate()
                # self.animate_epsilon()
        self.need_restart = False
        self.temps = 0
        self.episode += 1


    def clear_battleground(self):
        print("clear_battleground")
        # remove image, reference to the laser and the corresponding object from the battleground
        while self.images.lasers != []:
            self.ihm["carte"].delete(self.images.lasers[0])
            del self.images.lasers[0]
        self.battleground.lasers = []

        # remove image of the targets
        for i, target_images in enumerate(self.images.targets):
            self.ihm["carte"].delete(target_images)
            self.images.targets[i] = None

        # remove image and image reference of ships
        # but not the corresponding object from the battleground
        # they are kept as wreckage and repaired each game/restart/episode
        for i, ship_image in enumerate(self.images.ships):
            ship_image.destroy_image()
            # self.ihm["carte"].delete(ship_image)
            # self.images.ships[i] = None
        
        # reset the list of things to delete
        for key, value in self.todelete.items():
            self.todelete[key] = []

        return self.battleground.networks


    def link_functionnalities(self):
        """Link all commands/functionalities/features to their graphical component."""
        if "check_recording" in self.ihm:
            self.ihm["check_recording"].configure(command=self.swap_recording)
            self.master.bind("r", lambda e: self.ihm["check_recording"].invoke())
        if "train" in self.ihm:
            self.ihm["train"].configure(command=self.analyse_records)
        if "save_ia" in self.ihm:
            self.ihm["save_ia"].configure(command=self.save_ia)
        if "check_transfer_player" in self.ihm:
            self.ihm["check_transfer_player"].configure(command=self.transfer_player)
        # self.ihm["switch_session"].configure(command=self.create_switch_session)
        if "exploration" in self.ihm:
            self.ihm["exploration"].configure(command=self.actualise_exploration)
        if "restart" in self.ihm:
            self.ihm["restart"].configure(command=self.request_restart)

        # link functions to events
        # when the closing cross is pressed
        self.master.protocol("WM_DELETE_WINDOW", self.quit)

        self.ihm["vitesse"].set(3)
        self.ihm["check_continuous_training"].select()
        self.swap_continuous_training()


    def request_restart(self):
        self.need_restart = True

    def create_switch_session(self):
        Alert("New session", "Create", callback=lambda x : self.switch_session(x))


    def switch_session(self, name):
        path = os.path.join(NETWORKS_FOLDER, name)
        self.session_name = name
        os.makedirs(path, exist_ok=True)
        network = self.load_ia()
        # TODO when they will be more than one IA
        # if there is already an existing network we load it
        if network:
            self.battleground.set_ia(network)
            print("IA loaded")
        else:
            # else we create one
            if self.battleground.networks:
                self.save_ia()


    def transfer_player_ship(self):
        print("transfer")
        i = 0
        assigned = False
        while not assigned and i < len(self.battleground.ships):
            if self.battleground.ships[i].is_playable() and not self.battleground.ships[i].is_super_bot():
                player1 = Player("mouse", self.master, self.ihm["carte"])
                self.battleground.ships[i].assign_player(player1)
                if len(self.images.ships) > i:
                    self.images.ships[i].itemconfig(fill=self.battleground.ships[i].color)
                    # self.ihm["carte"].itemconfig(self.images.ships[i], fill=self.battleground.ships[i].color)
                assigned = True
            i += 1
        if not assigned:
            print("Impossible to assign a ship to player. None of them is playable.")


    def untransfer_player_ship(self):
        print("untransfer")
        for i, ship in enumerate(self.battleground.ships):
            if ship.player:
                print("unassign")
                ship.unassign_player()
                if len(self.images.ships) > i:
                    self.images.ships[i].itemconfig(fill=ship.color)
                    # self.ihm["carte"].itemconfig(self.images.ships[i], fill=ship.color)


    def save_ia(self):
        """There must be an IA to save in the battleground"""
        file = os.path.join(NETWORKS_FOLDER, self.session_name, "ofighter_network_" + now())
        self.battleground.networks[0].save(file)
        print("saving as ", file)


    def load_ia(self):
        """Return None if there is no network available."""
        network = None
        path = os.path.join(NETWORKS_FOLDER, self.session_name)
        files = glob.glob(os.path.join(path, "ofighter_network_*"))
        if files:
            # sorting by modification date : the more recent the firsts
            files.sort(key=os.path.getmtime, reverse=True)
            # for file in files:
            #     print("file", os.path.getmtime(file))
            network = Renforcement_learning_neural_network.load(files[0])
            print("loading", files[0])
        return network


    def save_all_ias(self):
        # TODO
        pass
        # path = os.path.join(NETWORKS_FOLDER, self.session_name, "ofighter_network_" + now())
        # print("saving as ", path)
        # nn.save(path)


    def swap_recording(self):
        if self.ihm["string_recording"].get() == "yes":
            self.recording = True
            self.record = OfighterRecord(self.battleground.absolute_state, len(self.battleground.ships), Battleground)
        elif self.ihm["string_recording"].get() == "no":
            self.save_records(os.path.join(RECORD_FOLDER, "ofighter_record_" + now()))
            self.records = None
            self.recording = False


    def transfer_player(self):
        if self.ihm["string_transfer_player"].get() == "yes":
            if self.ihm["string_training_mode"].get() == "no":
                self.transfer_player_ship()
        elif self.ihm["string_transfer_player"].get() == "no":
            if self.ihm["string_training_mode"].get() == "no":
                self.untransfer_player_ship()

    def save_records(self, name):
        self.record.save(name)

    def load_records(self, name):
        self.record = OfighterRecord.load(name)


    def swap_training_mode(self):
        if self.ihm["string_training_mode"].get() == "yes":
            if self.ihm["string_transfer_player"].get() == "yes":
                # the player do not need to play while the training mode is on
                self.untransfer_player_ship()
            if DISPLAY_EPSILONS and self.epsilon_graph.created:
                self.epsilon_graph.destroy()
            if DISPLAY_SCORES and self.score_graph.created:
                self.score_graph.destroy()
            if DISPLAY_LOSSES and self.loss_graph.created:
                self.loss_graph.destroy()
            if DISPLAY_ACTIONS_MAP and self.action_map_graph.created:
                self.action_map_graph.destroy()
            self.hide_map()

        elif self.ihm["string_training_mode"].get() == "no":
            self.expand_map()
            if self.ihm["string_transfer_player"].get() == "yes":
                # replace the player on the map if it was before
                self.transfer_player_ship()
            if DISPLAY_EPSILONS and not self.epsilon_graph.created:
                self.epsilon_graph.create()
            if DISPLAY_SCORES and not self.score_graph.created:
                self.score_graph.create()
            if DISPLAY_LOSSES and not self.loss_graph.created:
                self.loss_graph.create()
            if DISPLAY_ACTIONS_MAP and not self.action_map_graph.created:
                self.action_map_graph.create()


    def swap_is_learning(self):
        sb = self._super_bot_one is not None and hasattr(self._super_bot_one.agent, "is_learning")
        if self.ihm["string_is_learning"].get() == "yes":
            self.is_learning = True
            if sb :
                self._super_bot_one.agent.is_learning = True
        elif self.ihm["string_is_learning"].get() == "no":
            self.is_learning = False
            if sb :
                self._super_bot_one.agent.is_learning = False

    def analyse_records(self):
        # depreciated
        if "loading" in self.threads:
            print("Training already in progress !")
        else:
            # TODO be able to stop while in progress
            # TODO progress bar
            files = glob.glob(os.path.join(RECORD_FOLDER, "ofighter_record_*.orec"))
            # print("files len", len(files))
            # print("files", files)
            self.record = self.load_records(files)
            print("records len", len(self.record))
            print("analysing...")
            # make the loading bar appear
            self.ihm["progress"]["value"] = 0
            self.loading_i = 0
            self.ihm["progress"].grid(row=self.ihm["grid_raw_progress"], column=0)
            # the loading bar will load independantly
            self.read_loading()
            self.threads["loading"] = spawnthread(self.train_networks)


    def train_networks(self):
        # DEPRECIATED
        # update the neural network for all ships IAs
        for i, ship in enumerate(self.battleground.ships):
            ship.network.update_network(self.records["obs"], self.records["sol"])
            # print("{} of {}".format(i+1, len(self.battleground.ships)))
            self.loading_i = i
            sleep(0.2)
            # print("job done at", self.loading_i, "/", len(self.battleground.ships))
        # make the loading bar disappear
        self.loading_i = len(self.battleground.ships)
        self.ihm["progress"].grid_forget()
        self.threads["loading"].stop()
        del self.threads["loading"]


    def read_loading(self):
        print(self.battleground.ships[self.loading_i].network.progression)
        if self.loading_i == len(self.battleground.ships):
            self.ihm["progress"]["value"] = 100
        else:
            self.ihm["progress"]["value"] = round(
                100 / len(self.battleground.ships) * 
                (self.loading_i + self.battleground.ships[self.loading_i].network.progression)
                )

        print(self.ihm["progress"]["value"])

        if self.loading_i < len(self.battleground.ships)-1:
            self.master.after(100, self.read_loading)


    def actualise_exploration(self, value):
        # All bots share the same trainer so we only apply on one
        print("actualising exploration to ", value)
        if self._super_bot_one and hasattr(self._super_bot_one.agent, "trainer") and hasattr(self._super_bot_one.agent.trainer, "epsilon"):
            self._super_bot_one.agent.trainer.epsilon.set(float(value))


    def assign_ship_image(self, i, ship): # REPLACED
        # we create images for new ships
        self.images.ships[i] = self.ihm["carte"].create_oval(
                ship.body.x - ship.body.radius, ship.body.y - ship.body.radius,
                ship.body.x + ship.body.radius, ship.body.y + ship.body.radius,
                fill=ship.color, outline="Black", width="1"
            )


    def assign_target_image(self, i, ship):
        # we create image for new target associated to a ship
        size = 4
        # TODO crash if lines out of bounds ?
        self.images.targets[i] = self.ihm["carte"].create_line(
                ship.pointing.x - size, ship.pointing.y - size,
                ship.pointing.x + size, ship.pointing.y + size,
                ship.pointing.x - size, ship.pointing.y + size,
                ship.pointing.x + size, ship.pointing.y - size,
                fill="red",
                width="1"
            ) # fill=target.color

    def move_ship_image(self, i, ship): # REPLACED
        # print("state", ship.state)
        # move already existing image of existing ships
        self.ihm["carte"].coords(
            self.images.ships[i],
            ship.body.x - ship.body.radius, ship.body.y - ship.body.radius,
            ship.body.x + ship.body.radius, ship.body.y + ship.body.radius
        )

    def move_target_image(self, i, ship):
        # move already existing image of existing ship targets
        size = 4
        self.ihm["carte"].coords(
            self.images.targets[i],
            ship.pointing.x - size, ship.pointing.y - size,
            ship.pointing.x + size, ship.pointing.y + size,
            ship.pointing.x - size, ship.pointing.y + size,
            ship.pointing.x + size, ship.pointing.y - size,
        )

    def destroy_ship_image(self, i, ship): # REPLACED
        # explosion animation
        self.todelete["ships"].append((self.images.ships[i], self.battleground.ships[i]))
        self.images.ships[i] = None

    def destroy_target_image(self, i, ship):
        self.todelete["targets"].append(self.images.targets[i])
        self.images.targets[i] = None

    def actualise_ships(self):
        for i, ship in enumerate(self.battleground.ships):

            if self.recording:
                self.record.saveFrame(self.battleground.actions)

            if ship.actualise:
                self.images.ships[i].itemconfig(fill=ship.color)
                # self.ihm["carte"].itemconfig(self.images.ships[i], fill=self.battleground.ships[i].color)

            if ship.state == "wreckage":
                # image removed
                continue
            elif ship.state == "destroyed":
                # remove image
                self.images.ships[i].destroy_image()
                # self.destroy_ship_image(i, ship)
                # display the targeted area for the bot
                if ship.agent.behavior == "QlearnIA":
                    self.destroy_target_image(i, ship)
                ship.state = "wreckage"
                if ship.player and "check_transfer_player" in self.ihm :
                    self.ihm["check_transfer_player"].deselect()
                    # Leave the ship !
                    ship.unassign_player()
            elif self.temps >= 1 and ship.time == 0:
                # create image
                self.images.ships[i].assign_image(ship)
                # self.assign_ship_image(i, ship)
                # display the targeted area for the bot
                if ship.agent.behavior == "QlearnIA":
                    self.assign_target_image(i, ship)
            else:
                # move image
                self.images.ships[i].move_image(ship)
                # self.move_ship_image(i, ship)
                # display the targeted area for the bot
                if ship.agent.behavior == "QlearnIA":
                    self.move_target_image(i, ship)


    def actualise_lasers(self):
        for i, laser in enumerate(self.battleground.lasers):
            # print("type(laser.body.x)", type(laser.body.x))
            assert type(laser.body.x) in [int, float, np.float64]
            if laser.state == "destroyed":
                # explosion animation
                self.todelete["lasers"].append((self.images.lasers[i], self.battleground.lasers[i]))
            elif laser.time == 0:
                # we create images for new lasers
                self.images.lasers.append(
                    self.ihm["carte"].create_oval(
                        laser.body.x - laser.body.radius, laser.body.y - laser.body.radius,
                        laser.body.x + laser.body.radius, laser.body.y + laser.body.radius,
                        fill=laser.color
                    )
                )
            else:
                # and move already existing image of existing lasers
                self.ihm["carte"].coords(
                    self.images.lasers[i],
                    laser.body.x - laser.body.radius, laser.body.y - laser.body.radius,
                    laser.body.x + laser.body.radius, laser.body.y + laser.body.radius
                )


    def actualise_fps(self):
        delta = time() - self.start
        self.frames.append(delta)
        fps = 1 / mean(self.frames)
        # TODO running mean
        self.ihm["fps"]["text"] = "FPS {0:.0f}".format(fps)
        self.start = time()


    # TODO not adapted to launch multiple instances of Ofighters
    # musn't use decorators here
    # @fps(fps_manager)
    def frame(self):
        # print("Frame in thread : ", threading.current_thread().ident)

        self.temps += 1
        self.ihm["temps"]["text"] = "Temps : "+str(self.temps)
        self.ihm["episode"]["text"] = "Episode : "+str(self.episode)

        self.actualise_ships()
        self.actualise_lasers()
        # self.actualise_targets()
        # print("{} - {}".format(len(self.battleground.lasers), len(self.images.lasers)))
        self.clear_wreckage()

        if not self.training_mode:
            # sleep time goes from 2s to 1ms (0.5 fps to 1000fps)
            self.sleep_time = 1 / (10 ** self.ihm["vitesse"].get())

        if self._super_bot_one and hasattr(self._super_bot_one.agent, "trainer") and \
                hasattr(self._super_bot_one.agent.trainer, "epsilon") and \
                "exploration" in self.ihm:
            self.ihm["exploration"].setvar("exploration_value", self._super_bot_one.agent.trainer.epsilon.get())
            # print("DECAY TO ", self._super_bot_one.agent.trainer.epsilon.get())
            # print("position of bot", self._super_bot_one.body.x, self._super_bot_one.body.y)

        # if self.fps_manager.active:
        #     self.ihm["fps"]["text"] = "FPS " + str(self.fps_manager.fps)
        self.actualise_fps()

        if self.need_restart or self.continuous_training and self.temps > MAX_TIME:
            self.restart()
        else:
            # TODO must be first no ?
            self.battleground.frame()

        # print(measure_size)

        if self.quitter:
            self.master.quit()
            self.master.destroy()
        else:
            # print("after", int(1000 * self.sleep_time))
            self.master.after(int(1000 * self.sleep_time), self.frame)

        # sleep(self.sleep_time)


    def clear_wreckage(self):
        # remove image, reference to the laser and the corresponding object from the battleground
        for image, obj in self.todelete["lasers"]: # REPLACED
            self.ihm["carte"].delete(image)
            self.images.lasers.remove(image)
            self.battleground.lasers.remove(obj)

        # remove image, reference to the ship and the corresponding object from the battleground
        # for image, obj in self.todelete["ships"]:
        #     self.ihm["carte"].delete(image)
            # self.images.ships.remove(image)
            # self.battleground.ships.remove(obj)

        # remove image, reference to the target
        for image in self.todelete["targets"]:
            self.ihm["carte"].delete(image)
            # self.images.targets.remove(image)

        # the work is done
        for key, value in self.todelete.items():
            self.todelete[key] = []
        

    def quit(self):
        self.quitter = True
        # self.save_ia()
        if self.recording:
            self.save_records(os.path.join(RECORD_FOLDER, "ofighter_record_" + now()))


    """
    def animate_loss(self):
        # All bots share the same trainer so we only apply on one
        if self._super_bot_one and hasattr(self._super_bot_one.agent, "losses") and self.len_losses != len(self._super_bot_one.agent.losses):
            # print("animating in thread : ", threading.current_thread().ident)
            losses = self._super_bot_one.agent.losses
            self.len_losses = len(losses)
            self.ax_losses.cla()
            # print("losses", losses)

            self.ax_losses.plot(losses, label='Loss', linewidth=1.5)
            window_len = 11
            half_window_len = (window_len-1)//2
            if len(losses) >= window_len :
                # 'full', 'same', 'valid'
                losses_10 = np.convolve(np.array(losses), np.ones((window_len,))/window_len, mode='valid')
                # valid convolution loose border so we re-center
                x = range(half_window_len, half_window_len + len(losses_10))
                self.ax_losses.plot(x, losses_10, label='Loss on a {0} steps window'.format(window_len))
            self.ax_losses.legend(loc='upper left')
            plt.pause(0.5)

            # self.line_losses.set_data(range(len(losses)), losses) # set plot data
            # self.ax_losses.lines[0].set_data(range(len(losses)), losses) # set plot data
            # self.ax_losses.relim()                  # recompute the data limits
            # self.ax_losses.autoscale_view()         # automatic axis scaling
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()   # update the plot and take care of window events (like resizing etc.)
            # sleep(0.2)               # wait for next loop iteration

        self.master.after(500, self.animate_loss)


    def animate_score(self):
        # All bots share the same trainer so we only apply on one
        if self._super_bot_one and hasattr(self._super_bot_one.agent, "scores") and self.len_scores != len(self._super_bot_one.agent.scores):
            # print("animating in thread : ", threading.current_thread().ident)
            scores = self._super_bot_one.agent.scores
            self.len_scores = len(scores)
            self.ax_scores.cla()
            # print("scores", scores)

            self.ax_scores.plot(scores, label='Score', linewidth=1.5)
            window_len = 11
            half_window_len = (window_len-1)//2
            if len(scores) >= window_len :
                # 'full', 'same', 'valid'
                scores_10 = np.convolve(np.array(scores), np.ones((window_len,))/window_len, mode='valid')
                # valid convolution loose border so we re-center
                x = range(half_window_len, half_window_len + len(scores_10))
                self.ax_scores.plot(x, scores_10, label='Score on a {0} steps window'.format(window_len), color="green")
            self.ax_scores.legend(loc='upper left')
            plt.pause(0.5)
        # called at each restart instead
        # self.master.after(500, self.animate_score)


    def animate_epsilon(self):
        # All bots share the same trainer so we only apply on one
        if self._super_bot_one and hasattr(self._super_bot_one.agent, "epsilons") and self.len_epsilons != len(self._super_bot_one.agent.epsilons):
            # print("animating in thread : ", threading.current_thread().ident)
            epsilons = self._super_bot_one.agent.epsilons
            self.len_epsilons = len(epsilons)
            self.ax_epsilons.cla()
            # print("epsilons", epsilons)

            # x = range(len(epsilons))
            self.ax_epsilons.plot(epsilons, label='Epsilon', linewidth=1.5) # color="magenta"
            self.ax_epsilons.set_ylim(-0.1, 1.05)
            # if len(epsilons) > 10:
                # 'full', 'same', 'valid'
                # epsilons_10 = np.convolve(np.array(epsilons), np.ones((10,))/10, mode='same')
                # self.ax_epsilons.plot(epsilons_10, label='Epsilon on a 10 steps window', color="magenta")
            self.ax_epsilons.legend(loc='upper left')
            plt.pause(0.5)
        # called at each restart instead
        # self.master.after(500, self.animate_epsilon)

    
    def animate_actions_map(self):
        # All bots share the same trainer so we only apply on one
        if self._super_bot_one and hasattr(self._super_bot_one.agent, "trainer") and hasattr(self._super_bot_one.agent.trainer, "ptr_values")\
                and self._super_bot_one.agent.trainer.ptr_values is not None and self._super_bot_one.agent.trainer.act_values is not None:
            # TODO if different
            # print("animating in thread : ", threading.current_thread().ident)
            ptr_values = self._super_bot_one.agent.trainer.ptr_values
            act_values = self._super_bot_one.agent.trainer.act_values
            iaction =  np.argmax(act_values)
            act_values = np.expand_dims(act_values, axis=1)
            ipointer = np.unravel_index(np.argmax(ptr_values, axis=None), ptr_values.shape[0:2], order='F')

            print("--- ACTUALISATION ---")
            # print("prediction", act_values)
            # print("ptr_values", ptr_values.shape)
            ACTION = ["shoot", "thrust", "turn"]
            print("action", ACTION[iaction])
            # print("pointer", ipointer)

            self.ax_actions_map[0].cla()
            self.ax_actions_map[1].cla()
            self.img_ptr = self.ax_actions_map[0].imshow(ptr_values, cmap='coolwarm', interpolation='none', origin='upper') # vmin=-8, vmax=8
            self.ax_actions_map[0].plot(ipointer[0], ipointer[1], marker='x', markersize=10, markerfacecolor='r', markeredgecolor='black')
            self.img_act = self.ax_actions_map[1].imshow(act_values, cmap='OrRd', interpolation='none', origin='upper')#extent=[-1,1,-1,1], aspect=4)
            # self.ax_actions_map[0].axis('off')
            self.ax_actions_map[1].axis('off')

            if not self.colorbar_ptr:
                self.colorbar_ptr = self.fig_actions_map.colorbar(self.img_ptr, ax=self.ax_actions_map[0], extend='both')
                self.colorbar_act = self.fig_actions_map.colorbar(self.img_act, ax=self.ax_actions_map[1], extend='both')
            else:
                self.img_ptr.set_clim(vmin=ptr_values.min(),vmax=ptr_values.max())
                self.img_act.set_clim(vmin=act_values.min(),vmax=act_values.max())
                # self.colorbar_ptr.set_ticks(np.linspace(0., 2., num=11, endpoint=True))
                # self.colorbar_ptr.draw_all()

            # self.ax_actions_map.legend(loc='upper left')
            plt.pause(0.9)
        # TODO called at each frame instead ?
        self.master.after(100, self.animate_actions_map)
        """


    def run(self):
        self.master.after(0, self.frame)




if __name__ == "__main__":
    Ofighters()


