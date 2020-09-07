#!/usr/bin/python
# -*- coding: utf-8 -*-

from couple import Couple

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


if __name__ == "__main__":
    Player()