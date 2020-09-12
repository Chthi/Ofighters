#!/usr/bin/python
# -*- coding: utf-8 -*-

from random import randint, random, choice

from couple import Couple
from observation import Observation, DEFAULT_WIDTH, DEFAULT_HEIGHT
from ship import Ship


class Battleground():
    # TODO parameter to pass nb of IA of each type
    def __init__(self, state=None, ships=2, largeur=DEFAULT_WIDTH, hauteur=DEFAULT_HEIGHT, networks=[]):
        """Create a battleground with ships
        networks must be a list of size ship_number containing neural networks
        if not set the networks will be generated randomly
        ships : (int) the following number of ships will be created with default behavior
                (dict) {"behavior" : number} n ships of each behavior will be created
        """
        self.background = '#000000'

        default_behavior = "random" # "q_learning"
        if isinstance(ships, dict):
            self.ships_map = ships
            self.ships_number = len(ships)
        elif isinstance(ships, int):
            self.ships_number = ships
            self.ships_map = {default_behavior : ships}
        else:
            raise Exception("ships argument must be int or dict.")

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

        for behavior, number in self.ships_map.items():
            for n in range(number):
                self.ships.append(Ship(randint(0, largeur), randint(0, hauteur), self, behavior=behavior))

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
            ship.reset(x=randint(0, self.dim.x), y=randint(0, self.dim.y))

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
        # call everyone even wreckage
        # if is allowed to return None as action
        # in our case it will be interpreted as no action (wreckage typically)
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



if __name__ == "__main__":
    Battleground().run()
