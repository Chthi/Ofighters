from __future__ import print_function
import threading
import time
import queue


class ThreadedClient(threading.Thread):
    def __init__(self, qe, fcn):
        super(ThreadedClient, self).__init__()
        self.iterations = 0
        self.daemon = True  # Allow main to exit even if still running.
        self.paused = True  # Start out paused.
        self.state = threading.Condition()
        self.quit = False
        self.qe = qe
        self.fcn = fcn

    def run(self):
        self.resume()
        while not self.quit:
            with self.state:
                if self.paused:
                    self.state.wait()  # Block execution until notified.
                else:
                    self.qe.put(self.fcn())
            # Do stuff.
            time.sleep(0.1)
            self.iterations += 1

    def resume(self):
        with self.state:
            self.paused = False
            self.state.notify()  # Unblock self if waiting.
            print("resuming...")

    def pause(self):
        with self.state:
            self.paused = True  # Block self.
            print("pausing...")

    def stop(self):
        with self.state:
            self.quit = True
            print("quitting...")


class Stopwatch(object):
    """ Simple class to measure elapsed times. """
    def start(self):
        """ Establish reference point for elapsed time measurements. """
        self.start_time = time.time()
        return self.start_time

    @property
    def elapsed_time(self):
        """ Seconds since started. """
        try:
            start_time = self.start_time
        except AttributeError:  # Wasn't explicitly started.
            start_time = self.start()

        return time.time() - start_time


def spawnthread(fcn):
    thread = ThreadedClient(qe, fcn)
    thread.start()
    return thread


qe = queue.Queue()


if __name__ == '__main__':
    MAX_RUN_TIME = 5  # Seconds.
    concur = ThreadedClient()
    stopwatch = Stopwatch()

    print('Running for {} seconds...'.format(MAX_RUN_TIME))
    concur.start()
    while stopwatch.elapsed_time < MAX_RUN_TIME:
        concur.resume()
        # ... do some concurrent operations.
        concur.pause()
        # Do some other stuff...
    concur.stop()

    # Show ThreadedClient thread executed.
    print('concur.iterations: {}'.format(concur.iterations))