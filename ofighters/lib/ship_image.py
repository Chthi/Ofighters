#!/usr/bin/python
# -*- coding: utf-8 -*-


class ShipImage():

    def __init__(self, ihm, battleground):
        # self.id = id
        self.ihm = ihm
        self.battleground = battleground

        self.image = None

        self.todelete = []


    def itemconfig(self, *args, **kwargs):
        # edit
        self.ihm["carte"].itemconfig(self.image, *args, **kwargs)


    def assign_image(self, ship):
        # we create images for new ships
        self.image = self.ihm["carte"].create_oval(
                ship.body.x - ship.body.radius, ship.body.y - ship.body.radius,
                ship.body.x + ship.body.radius, ship.body.y + ship.body.radius,
                fill=ship.color, outline="Black", width="1"
            )


    def move_image(self, ship):
        # move already existing image of existing ships
        self.ihm["carte"].coords(
            self.image,
            ship.body.x - ship.body.radius, ship.body.y - ship.body.radius,
            ship.body.x + ship.body.radius, ship.body.y + ship.body.radius
        )


    def destroy_image(self):
        # explosion animation
        # as the image reference and the object are kept
        # we can delete image directly when destroyed
        self.ihm["carte"].delete(self.image)
        self.image = None

