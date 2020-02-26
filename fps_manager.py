#!/usr/bin/python
# -*- coding: utf-8 -*-

from time import sleep, time


class fps_manager():

    def __init__(self, refresh=2):
        self.last_fps_refresh_time = time()
        self.active = True
        self.refresh_time = refresh
        self.run_time = 0
        self.size = 0
        self.fps = 1

    def give(self, start, end):
        if self.active:
            run_time = end - start
            # we refresh run_time list every x second
            # so fps count is more precise at the end of the time x
            # but it would be way more expensive to rotate the list
            # based on the arrival time of the run_time
            if end - self.last_fps_refresh_time > self.refresh_time:
                self.last_fps_refresh_time = end
                self.run_time = 0
                self.size = 0
            self.run_time += run_time
            self.size += 1
            self.fps = round(1 / (self.run_time / self.size), 1)


def fps(manager):
    def with_manager(method):
        def with_fps(*args, **keyargs):
            start = time()
            result = method(*args, **keyargs)
            end = time()
            manager.give(start, end)
            return result
        return with_fps
    return with_manager



def test_loop():
    start = time()
    for i in range(1000):
        sleep(0.001)
    return time() - start


if __name__ == '__main__':
    fpsm = fps_manager()

    @fps(fpsm)
    def do_things():
        sleep(0.001)


    def test_fps_mananger():
        for i in range(1000):
            do_things()


    test_fps_mananger()

    print(fpsm.fps, "fps")

    loop_slowing = test_loop()
    manager_slowing = 1000 / fpsm.fps

    print("loop_slowing", loop_slowing)
    print("manager_slowing", manager_slowing)
    print("slowing ratio {}%".format(((manager_slowing / loop_slowing) - 1) * 100))