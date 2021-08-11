#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle


class OfighterRecord():

    def __init__(self, observation, nb_agents, game_cls, *args, **kwargs):
        # TODO generalisation using
        self.initial_state = observation
        self.nb_agents = nb_agents
        self.game_cls = game_cls

        self.args = args
        self.kwargs = kwargs

        # list of actions for each frame
        self.actions = []

        self.rewind()

    def saveFrame(self, actions):
        self.actions.append(actions)


    def nextFrame(self):
        """Generate the next observation using a given action for all agents in this one.
        simplified example :
            state = 0 and action = +1 will give state = 1
        The game must be deterministic (or have a fixed random seed)"""
        self.game.frame(self.actions[self.reading_head])
        self.reading_head += 1
        return self.game.state


    def rewind(self):
        self.reading_head = 0
        self.game = self.game_cls(self.initial_state, self.nb_agents, *self.args, **self.kwargs)


    def __str__(self):
        return str(self.actions)


    def save(self, name):
        """save the record without the game object. reset the reading"""
        print("saving {0} frames...".format(len(self.actions)))
        self.reading_head = 0
        self.game = None
        if self.actions:
            if not name.endswith(".orec"):
                name += ".orec"
            with open(name, "wb+") as file:
                pickle.dump(self, file)


    @classmethod
    def load(self, name):
        """Load the record and initialise the reading."""
        if not name.endswith(".orec"):
            name += ".orec"
        with open(name , "rb") as file:
            record = pickle.load(file)
        record.rewind()
        return record


    def load(self, paths):
        raise Exception("Not implemented")
        """Load the record and initialise the reading."""
        if not isinstance(paths, list):
            paths = [paths]
        for path in paths:
            if not path.endswith(".orec"):
                path += ".orec"
            with open(path , "rb") as file:
                content = pickle.load(file)
                if isinstance(content, dict):
                    # print(content)
                    records["obs"].extend(content["obs"])
                    records["sol"].extend(content["sol"])
                else:
                    raise Exception("Bad file content : dict expected")



class Game():
    """Simple game for illustration and test.
    You can imagine a race between 2 guys"""
    def __init__(self, obs, agts, arg, kwarg):
        self.state = [0, 0]

    def frame(self, actions):
        for i in range(len(self.state)):
            self.state[i] += actions[i]


if __name__ == "__main__" :
    record = OfighterRecord([0,0], 2, Game, "arg", kwarg="kwarg")
    record.saveFrame([1, 0])
    record.saveFrame([2, 2])
    record.saveFrame([1, 1])
    record.saveFrame([0, 2])
    print(record)

    obs = record.nextFrame()
    print(obs)
    obs = record.nextFrame()
    print(obs)
    obs = record.nextFrame()
    print(obs)
    obs = record.nextFrame()
    print(obs)


