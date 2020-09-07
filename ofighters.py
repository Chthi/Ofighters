#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple, deque
from copy import copy, deepcopy

from time import sleep, time
import tkinter as tk
import numpy as np
import datetime
import pickle
import json
import glob
import threading
from itertools import count

from record import OfighterRecord
from thread_manager import spawnthread
from couple import Couple
from form import Circle
from player import Player
from battleground import Battleground
from laser import Laser
from action import Action
from observation import Observation, DEFAULT_WIDTH, DEFAULT_HEIGHT
# from brAIn import NETWORKS_FOLDER, BrAIn, SimpleModel
NETWORKS_FOLDER = "ofighter_networks"

from map_menu_struct import *
from fps_manager import fps, fps_manager

# from renforcement_learning_neural_network import Renforcement_learning_neural_network

import matplotlib
# While GTK isn't avail everywhere, we use TkAgg backend to generate png
if sys.platform != "win32" and os.getenv("DISPLAY") is None :
    matplotlib.use("Agg")
else :
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


SHIPS_NUMBER = 7

MAP_PRECISION = 4

MAX_TIME = 200
ANTICIPATION = 100

RECORD_FOLDER = "ofighter_records"

DISPLAY_LOSSES = True

def now():
    return datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")



# coordinates conventions
#   | 0 | 1 | 2 | 3 | 4 -> x
# 0 |
# 1 |          pi/2
# 2 |           |
# 3 |   (-)pi _   _ 0
# 4 |           |
# y            -pi/2



class Ofighters(MapMenuStruct):
    fps_manager = fps_manager()

    def __init__(self, largeur=DEFAULT_WIDTH, hauteur=DEFAULT_HEIGHT):
        # we call the basic graphic interface for the game
        super().__init__(largeur, hauteur)

        self.session_name = ""
        # if we record the player ingame
        self.recording = False
        # self.training_mode = False
        self.record = None
        # contains all the objects we will need to delete
        self.todelete = {}

        plt.style.use('fivethirtyeight')

        self.battleground = Battleground(ship_number=SHIPS_NUMBER, largeur=self.dim.x, hauteur=self.dim.y)
        self.switch_session("default")

        Images = namedtuple("Images", ['ships', 'lasers', ])
        ships_images_list = len(self.battleground.ships) * [None]
        self.images = Images(ships_images_list, [], )

        # images contains tkinter graphical objects
        # lasers contains (index, object) indexes of ships images
        # and corresponding object in the battleground
        # lasers contains indexes of ships images
        self.todelete = {"images" : [], "lasers" : [], "ships" : []}

        self.threads = {}

        # main window
        self.master = tk.Tk()

        # fill the main window
        self.create_main_window()

        # linking Ofighters functions to the main graphical structure
        self.link_functionnalities()

        # self.create_widgets()
        self.player_ship = True
        self.transfer_player_ship()

        print("Main thread : ", threading.current_thread().ident)
        # other windows
        self.run()
        # self.master.after(0, self.run)
        # self.threads["run"] = spawnthread(self.run)

        # losses graphic window
        if DISPLAY_LOSSES:
            # plt.ion()
            self.fig = plt.figure()
            self.ax_losses = self.fig.add_subplot(1,1,1)
            # self.ax_losses = plt.subplot(111)
            self.ax_losses.set_xlabel('Replays')
            self.ax_losses.set_ylabel('Loss')
            self.len_losses = 1
            # self.line_losses, = self.ax_losses.plot([0], [0])
            plt.tight_layout()
            self.master.after(1000, self.animate_loss)
            # self.master.after(0, plt.show)
            self.master.after(0, self.fig.show)
            print("showed")

        # run the main thread/loop
        print("mainloop")
        self.master.mainloop()

    def add_plot(self):
        x_vals = [0, 1, 2, 3, 4, 5]
        y_vals = [0, 1, 3, 2, 3, 5]
        plt.plot(x_vals, y_vals)
        plt.show()

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
        self.temps = 0
        self.episode += 1


    def clear_battleground(self):
        print("clear_battleground")
        # remove image, reference to the laser and the corresponding object from the battleground
        while self.images.lasers != []:
            self.ihm["carte"].delete(self.images.lasers[0])
            del self.images.lasers[0]
        self.battleground.lasers = []
        
        # remove image and image reference of ships
        # but not the corresponding object from the battleground
        # they are kept as wreckage and repaired each game/restart/episode
        for i, ship_image in enumerate(self.images.ships):
            self.ihm["carte"].delete(ship_image)
            self.images.ships[i] = None
        
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
            if self.battleground.ships[i].is_playable():
                player1 = Player("mouse", self.master, self.ihm["carte"])
                self.battleground.ships[i].assign_player(player1)
                if len(self.images.ships) > i:
                    self.ihm["carte"].itemconfig(self.images.ships[i], fill=self.battleground.ships[i].color)
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
                    self.ihm["carte"].itemconfig(self.images.ships[i], fill=ship.color)


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
            self.expand_map()
        elif self.ihm["string_training_mode"].get() == "no":
            self.hide_map()
            if self.ihm["string_transfer_player"].get() == "yes":
                # replace the player on the map if it was before
                self.transfer_player_ship()


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
        if hasattr(self.battleground.ships[0].agent, "trainer") and hasattr(self.battleground.ships[0].agent.trainer, "epsilon"):
            self.battleground.ships[0].agent.trainer.epsilon = float(value)


    def assign_ship_image(self, i, ship):
        # we create images for new ships
        self.images.ships[i] = self.ihm["carte"].create_oval(
                ship.body.x - ship.body.radius, ship.body.y - ship.body.radius,
                ship.body.x + ship.body.radius, ship.body.y + ship.body.radius,
                fill=ship.color, outline="Black", width="1"
            )

    def move_ship_image(self, i, ship):
        # print("state", ship.state)
        # and move already existing image of existing ships
        self.ihm["carte"].coords(
            self.images.ships[i],
            ship.body.x - ship.body.radius, ship.body.y - ship.body.radius,
            ship.body.x + ship.body.radius, ship.body.y + ship.body.radius
        )

    def destroy_ship_image(self, i, ship):
        # explosion animation
        self.todelete["ships"].append((self.images.ships[i], self.battleground.ships[i]))
        self.images.ships[i] = None


    def actualise_ships(self):
        for i, ship in enumerate(self.battleground.ships):

            if ship.actualise:
                self.ihm["carte"].itemconfig(self.images.ships[i], fill=self.battleground.ships[i].color)

            if self.recording:
                self.record.saveFrame(self.battleground.actions)

            if ship.state == "wreckage":
                # image removed
                continue
            elif ship.state == "destroyed":
                # remove image
                self.destroy_ship_image(i, ship)
                ship.state = "wreckage"
                if ship.player and "check_transfer_player" in self.ihm :
                    self.ihm["check_transfer_player"].deselect()
                    # Leave the ship !
                    ship.unassign_player()
            elif self.temps >= 1 and ship.time == 0:
                # create image
                self.assign_ship_image(i, ship)
            else:
                # move image
                self.move_ship_image(i, ship)


    def actualise_lasers(self):
        for i, laser in enumerate(self.battleground.lasers):
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
        # print("{} - {}".format(len(self.battleground.lasers), len(self.images.lasers)))
        self.clear_wreckage()

        if not self.training_mode:
            # sleep time goes from 2s to 1ms (0.5 fps to 1000fps)
            self.sleep_time = 1 / (10 ** self.ihm["vitesse"].get())

        if hasattr(self.battleground.ships[0].agent, "trainer") and \
                hasattr(self.battleground.ships[0].agent.trainer, "epsilon") and \
                "exploration" in self.ihm:
            self.ihm["exploration"].set(self.battleground.ships[0].agent.trainer.epsilon)
            # print("DECAY TO ", self.battleground.ships[0].agent.trainer.epsilon)

        # All bots share the same trainer so we only apply on one
        # if hasattr(self.battleground.ships[0].agent, "losses"):
        #     self.animate_loss(self.battleground.ships[0].agent.losses)

        # if self.fps_manager.active:
        #     self.ihm["fps"]["text"] = "FPS " + str(self.fps_manager.fps)

        if self.continuous_training and self.temps > MAX_TIME:
            self.restart()
        else:
            # TODO must be first no ?
            self.battleground.frame()

        if self.quitter:
            self.master.quit()
            self.master.destroy()
        else:
            # print("after", int(1000 * self.sleep_time))
            self.master.after(int(1000 * self.sleep_time), self.frame)

        # sleep(self.sleep_time)


    def clear_wreckage(self):
        # remove image, reference to the laser and the corresponding object from the battleground
        for image, obj in self.todelete["lasers"]:
            self.ihm["carte"].delete(image)
            self.images.lasers.remove(image)
            self.battleground.lasers.remove(obj)

        # remove image, reference to the ship and the corresponding object from the battleground
        for image, obj in self.todelete["ships"]:
            self.ihm["carte"].delete(image)
            # self.images.ships.remove(image)
            # self.battleground.ships.remove(obj)
        
        # the work is done
        for key, value in self.todelete.items():
            self.todelete[key] = []
        

    def quit(self):
        self.quitter = True
        # self.save_ia()
        if self.recording:
            self.save_records(os.path.join(RECORD_FOLDER, "ofighter_record_" + now()))


    # self.loss_index = count()
    def animate_loss(self):
        if not self.quitter:
            # All bots share the same trainer so we only apply on one
            if hasattr(self.battleground.ships[0].agent, "losses") and self.len_losses != len(self.battleground.ships[0].agent.losses):
                # print("animating in thread : ", threading.current_thread().ident)
                losses = self.battleground.ships[0].agent.losses
                self.len_losses = len(losses)
                self.ax_losses.cla()
                print("losses", losses)

                # x = range(len(losses))
                self.ax_losses.plot(losses, label='Loss', linewidth=1.5)
                if len(losses) > 5:
                    # 'full', 'same', 'valid'
                    losses_5 = np.convolve(np.array(losses), np.ones((5,))/5, mode='same')
                    self.ax_losses.plot(losses_5, label='Loss on a 5 steps window')
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


    def run(self):
        self.master.after(0, self.frame)





# TODO benchmark desactivate print
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')
# def enablePrint():
#     sys.stdout = sys.__stdout__


if __name__ == "__main__":
    Ofighters()


