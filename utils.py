
import datetime

DEBUG = True

def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def now():
    return datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")



if __name__ == "__main__":
    debug("Hello", "debug")
    print(now())
    debug(now())