#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import math
import functools

from random import randint, random, choice
from collections import namedtuple, deque
from copy import copy, deepcopy

from time import sleep, time
import tkinter as tk
import numpy as np
import datetime
import pickle
import json
import glob

from record import OfighterRecord
from thread_manager import spawnthread
from couple import Couple
from form import Circle
from ship import Ship
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

# TODO display a graph of rewards per minutes
# TODO button to remove leroy jenkins
# TODO stats actions, score, plot

# TODO keybind r to record

SHIPS_NUMBER = 7

MAP_PRECISION = 4

MAX_TIME = 200
ANTICIPATION = 100


RECORD_FOLDER = "ofighter_records"


# TODO kill when shooting inside opponent ?
# TODO storage of observations huge. use a generative aproch for records ? ie. only record actions and replay the match
# TODO homogenise fps (no wait if computation time high etc)
# TODO store all the actions an then play them all
# TODO load network that match required layers/token/unique string
# TODO there is a slight possibility to kill yourself (maybe when shooting and moving during the same frame)

# TODO sleep less each frame is compute time is high (but set minimum sleep)
# TODO add laser sight to player ship
# TODO statistic object containing all kills, etc..
# TODO different type of ships
# TODO different type of thrusters
# TODO evolving ships design
# TODO evolving ships ia
# TODO scripted evolving ships ia
# TODO add a button to see what the neural network see
# TODO increase the max processor usage (multitheading ?)
# TODO add pad click 1 with mouse click one
# TODO add a restart button

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



class Player():
    max_id = 1
    keysets = {
        "mouse" : {
            "<Button-1>" : "shoot", 
            "a" : "thrust", 
            "<Motion>" : "pointing",
        }
    }

    def __init__(self, keyset, master, carte):
        # TODO automatized named tuple
        self.id = Player.max_id
        Player.max_id += 1
        self.keyset = Player.keysets[keyset]
        # for event, tAction in self.keyset.items():
            # carte.bind(event, tAction[1])
        carte.bind("<Button-1>", lambda e : self.press_shoot(e))
        carte.bind("<ButtonRelease-1>", lambda e : self.unpress_shoot(e))
        master.bind("a", lambda e : self.request_thrust(e))
        master.bind("<Motion>", lambda e : self.request_turn(e))
        self.shoot = False
        self.clear_keys()


    def press_shoot(self, event):
        self.actions_set.add("shoot")
        # print(self.actions_set)
        self.shoot = True
        # print("request shoot")

    def unpress_shoot(self, event):
        # print(self.actions_set)
        if "shoot" in self.actions_set:
            self.actions_set.remove("shoot")
        self.shoot = False
        # print("request cease fire")

    def request_thrust(self, event):
        self.actions_set.add("thrust")
        self.thrust = True
        # print("request thrust")

    def request_turn(self, event):
        self.actions_set.add("pointing")
        self.turn = True
        self.cursor = Couple(event.x, event.y)
        # print("request turn", self.cursor)

    def clear_keys(self):
        self.actions_set = set()
        if self.shoot:
            self.actions_set.add("shoot")
        self.thrust = False
        self.turn = False
        self.cursor = None



class Battleground():
    # TODO parameter to pass nb of IA of each type
    def __init__(self, state=None, ship_number=2, largeur=DEFAULT_WIDTH, hauteur=DEFAULT_HEIGHT, networks=[]):
        """Create a battleground with ships
        networks must be a list of size ship_number containing neural networks
        if not set the networks will be generated randomly"""
        self.background = '#000000'
        self.ships_number = ship_number
        self.time = 0
        self.dim = Couple(largeur, hauteur)
        self.ships = []
        self.lasers = []
        self.networks = networks
        # action vector of each ship at each given moment/frame
        self.actions = []

        # list of all the rewards get and there obtention time
        # ex : (1, 120)
        self.last_x_time_rewards = []
        # expiration time of rewards in this list
        self.reward_list_len = 100

        # for the moment all the ships share the same network.
        # it's faster to train
        # if self.networks == []:
            # TODO import model created
            # self.networks = [network]

        # if not networks:
        #     networks = []
        #     for i in range(self.ships_number):
        #         network = Renforcement_learning_neural_network(
        #                     [Observation.size, 9, Action.size], 
        #                     history_size=ANTICIPATION)
        #         # we force a bit the shoot and thrust actions
        #         # because spinning ships aren't fun
        #         network.biases_layers[-1][1][0] = 3 * abs(network.biases_layers[-1][1][0])
        #         network.biases_layers[-1][2][0] = 3 * abs(network.biases_layers[-1][2][0])
        #         for i in range(len(network.biases_layers[-1])):
        #             network.biases_layers[-1][i][0] = network.biases_layers[-1][i][0] / 4
        #         networks.append(network)

        # copy by reference so at the end
        # when the ships will be deleted the networks will stay
        # self.networks = networks

        """
        if len(self.networks) == 1:
            for x in range(self.ships_number):
                self.ships.append(Ship(randint(0, largeur), randint(0, hauteur), self, self.networks[0]))
        else:
            for x in range(self.ships_number):
                self.ships.append(Ship(randint(0, largeur), randint(0, hauteur), self, self.networks[x]))
        """

        for x in range(self.ships_number):
            # network random
            # self.ships.append(Ship(randint(0, largeur), randint(0, hauteur), self, behavior="network"))
            self.ships.append(Ship(randint(0, largeur), randint(0, hauteur), self, behavior="q_learning"))
            # self.ships.append(Ship(randint(0, largeur), randint(0, hauteur), self, behavior="random"))

        # print("there")
        # self.time_list = [self.time]
        # # list of average rewards
        # self.acc_rewards = [0]
        # # Creation of the graph of avg reward
        # fig, ax = plt.subplots()
        # # Size of the graph
        # plt.axis([0, 1, 0, 50])
        # # real time modification handeling
        # plt.ion()
        # plt.title("Summed rewards of last frames")
        # plt.xlabel("time")
        # plt.ylabel("avg rewards last {} frames".format(self.reward_list_len))
        # plt.xlim(32, 212)
        # plt.grid(True)
        # self.points, = ax.plot(self.time_list, self.acc_rewards, marker='o', linestyle='-')

        if state:
            self.absolute_state = state
            Observation.loadBattleground(self, state)
        else:
            self.absolute_state = Observation(battleground=self)


    def restart(self):
        self.time = 0
        self.lasers = []
        self.actions = []

        for ship in self.ships:
            # print("RESET")
            ship.reset()

        self.absolute_state = Observation(battleground=self)


    def set_ia(self, network):
        for ship in self.ships:
            ship.network = network


    def outside(self, x, y):
        return (x < 0) or (y < 0) or (x >= self.dim.x) or (y >= self.dim.y)


    # def plot_last_time_rewards(self):
    #     # temps.append(self.time)
    #     self.acc_rewards.append(sum(map(lambda x:x[0], self.last_x_time_rewards)))
    #     self.acc_rewards.append(self.time)

    #     # On recentre le graphique
    #     plt.axis([0, self.time_list[-1] - self.time_list[0] + 1, 0, max(self.acc_rewards)+1])
        
    #     # on place le nouveau point
    #     # plt.scatter(self.time, data)
    #     self.points.set_data(self.time_list, self.acc_rewards)

    #     # if the oldest recorded reward passed his expiration date we remove it
    #     if self.last_x_time_rewards and self.time - self.last_x_time_rewards[0][0] > self.reward_list_len :
    #         self.last_x_time_rewards.pop()


    def request_actions(self):
        return [ship.get_action(self.absolute_state) for ship in self.ships]


    def generate_frame(self, actions):
        # self.plot_last_time_rewards()
        self.time += 1
        # TODO compute simultaneously so first ships don't have advantage on last ones
        for laser in self.lasers:
            laser.move()
        for i, ship in enumerate(self.ships):
            ship.move(actions[i])


    def frame(self):
        self.actions = self.request_actions()
        self.generate_frame(self.actions)
        self.absolute_state = Observation(battleground=self)


    def run(self):
        # used without interface so ctrl+C is quit for the moment
        # TODO improve quitting, but not used anyway
        while 1:
            self.frame()



class Ofighters(MapMenuStruct):
    fps_manager = fps_manager()

    def __init__(self, size=20, largeur=DEFAULT_WIDTH, hauteur=DEFAULT_HEIGHT):
        # we call the basic graphic interface for the game
        super().__init__(largeur, hauteur)

        self.session_name = ""
        # if we record the player ingame
        self.recording = False
        # self.training_mode = False
        self.record = None
        # contains all the objects we will need to delete
        self.todelete = {}

        self.battleground = Battleground(ship_number=SHIPS_NUMBER, largeur=self.dim.x, hauteur=self.dim.y)
        self.switch_session("default")

        Images = namedtuple("Images", ['ships', 'lasers', ])
        self.images = Images([], [], )
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

        # other windows
        self.threads["run"] = spawnthread(self.run)
        
        # run the main thread/loop
        self.master.mainloop()



    def restart(self):
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
        self.battleground.restart()
        # self.battleground = Battleground(ship_number=SHIPS_NUMBER, hauteur=self.dim.x, largeur=self.dim.y)
        # we put back the player ship on the field if required
        # not used anymore as the ships objects are not deleted and the player stays in the ship until destroyed
        # if self.ihm["string_transfer_player"].get() == "yes" and self.ihm["string_training_mode"].get() == "no":
        #     self.transfer_player_ship()
        self.temps = 0
        self.episode += 1


    def clear_battleground(self):

        # remove image, reference to the laser and the corresponding object from the battleground
        while self.images.lasers != []:
            self.ihm["carte"].delete(self.images.lasers[0])
            del self.images.lasers[0]
        # self.battleground.lasers = []
        
        # remove image, reference to the ship and the corresponding object from the battleground
        while self.images.ships != []:
            self.ihm["carte"].delete(self.images.ships[0])
            del self.images.ships[0]
        # self.battleground.ships = []
        
        # the work is done
        for key, value in self.todelete.items():
            self.todelete[key] = []

        return self.battleground.networks


    def link_functionnalities(self):
        self.ihm["check_recording"].configure(command=self.swap_recording)
        self.master.bind("r", lambda e: self.ihm["check_recording"].invoke())
        self.ihm["train"].configure(command=self.analyse_records)
        self.ihm["save_ia"].configure(command=self.save_ia)
        self.ihm["check_transfer_player"].configure(command=self.transfer_player)
        self.ihm["switch_session"].configure(command=self.create_switch_session)
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


    # TODO pause
    # TODO click on ship to save IA (Save this IA)
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

    # TODO max size on obs/act vectors un renforcement nn
    # TODO multiplayer ? :)

    def save_records(self, name):
        self.record.save(name)

    def load_records(self, name):
        self.record = OfighterRecord.load(name)


    def swap_training_mode(self):
        # TODO replace all the things at the right place (use grid ?)
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


    # TODO not adapted to launch multiple instances of Ofighters
    # musn't use decorators here
    @fps(fps_manager)
    def frame(self):

        for i, ship in enumerate(self.battleground.ships):

            if ship.actualise:
                self.ihm["carte"].itemconfig(self.images.ships[i], fill=self.battleground.ships[i].color)

            if self.recording:
                self.record.saveFrame(self.battleground.actions)

            if ship.state == "wreckage":
                continue
            elif ship.state == "destroyed":
                # explosion animation
                # self.todelete.append(lambda : self.ihm["carte"].delete(self.images.ships[i]) )
                # self.todelete["images"].append(self.images.ships[i])
                # self.todelete["objects"].append(self.battleground.ships[i])
                # self.todelete["ships"].append(self.images.ships[i])

                self.todelete["ships"].append((self.images.ships[i], self.battleground.ships[i]))
                # self.ihm["carte"][i] = None
                self.images.ships[i] = None
                ship.state = "wreckage"
                if ship.player:
                    self.ihm["check_transfer_player"].deselect()
                    # Leave the ship !
                    ship.unassign_player()

            elif self.temps > 1 and ship.time == 0:
                # we create images for new ships
                self.images.ships[i] = self.ihm["carte"].create_oval(
                        ship.body.x - ship.body.radius, ship.body.y - ship.body.radius,
                        ship.body.x + ship.body.radius, ship.body.y + ship.body.radius,
                        fill=ship.color, outline="Black", width="1"
                    )
            elif self.temps == 1 and ship.time == 0:
                # we create images for new ships
                self.images.ships.append(
                    self.ihm["carte"].create_oval(
                        ship.body.x - ship.body.radius, ship.body.y - ship.body.radius, 
                        ship.body.x + ship.body.radius, ship.body.y + ship.body.radius, 
                        fill=ship.color, outline="Black", width="1"
                    )
                )
            else:
                # print("state", ship.state)
                # and move already existing image of existing ships
                self.ihm["carte"].coords(
                    self.images.ships[i],
                    ship.body.x - ship.body.radius, ship.body.y - ship.body.radius, 
                    ship.body.x + ship.body.radius, ship.body.y + ship.body.radius
                )


        for i, laser in enumerate(self.battleground.lasers):
            if laser.state == "destroyed":
                # explosion animation
                # self.todelete.append(lambda : self.ihm["carte"].delete(self.images.lasers[i]) )
                # self.todelete["images"].append(self.images.lasers[i])
                # self.todelete["objects"].append(self.battleground.lasers[i])
                # self.todelete["lasers"].append(self.images.lasers[i])
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
        # print("{} - {}".format(len(self.battleground.lasers), len(self.images.lasers)))


        # self.ihm["carte"].coords(self.images.mass_center[0], self.battleground.center_of_mass.x-ship.body.radius, self.battleground.center_of_mass.y-ship.body.radius, self.battleground.center_of_mass.x+ship.body.radius, self.battleground.center_of_mass.y+ship.body.radius)

        # self.ihm["carte"].coords(self.images.best_pos[0], self.battleground.best_pos.x-ship.body.radius, self.battleground.best_pos.y-ship.body.radius, self.battleground.best_pos.x+ship.body.radius, self.battleground.best_pos.y+ship.body.radius)

        self.clear_wreckage()

        if not self.training_mode:
            self.sleep_time = 1 / self.ihm["vitesse"].get()


        if hasattr(self.battleground.ships[0].agent, "trainer") and hasattr(self.battleground.ships[0].agent.trainer, "epsilon"):
            self.ihm["exploration"].set(self.battleground.ships[0].agent.trainer.epsilon)
            print("DECAY TO ", self.battleground.ships[0].agent.trainer.epsilon)




        sleep(self.sleep_time)

        self.battleground.frame()
        # TODO must be first no ?


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


    def run(self):
        # images contains tkinter graphical objects
        # lasers contains (index, object) indexes of ships images 
        # and corresponding object in the battleground
        # lasers contains indexes of ships images
        self.todelete = {"images" : [], "lasers" : [], "ships" : []}

        while not self.quitter:

            self.temps += 1
            self.ihm["temps"]["text"] = "Temps : "+str(self.temps)

            self.ihm["episode"]["text"] = "Episode : "+str(self.episode)

            self.frame()

            if self.fps_manager.active:
                self.ihm["fps"]["text"] = "FPS " + str(self.fps_manager.fps)

            if self.continuous_training and self.temps > MAX_TIME:
                self.restart()

        self.master.destroy()



# TODO benchmark desactivate print
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')
# def enablePrint():
#     sys.stdout = sys.__stdout__



# Battleground().run()

Ofighters()


