""" This function returns a string with the time elapsed between starttime,endtime """
from typing import Callable

import time

# pylint: disable=C0103

def time_fn(*arg_pos_or_keys, prefix: str = "", display_arg_name: bool = False) -> Callable:
    global outer_dec

    def outer_dec(func: Callable) -> Callable:
        global wrapper

        def wrapper(*args, **kwargs) -> Callable:
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            args_str = ""
            for pos_or_key in arg_pos_or_keys:
                if isinstance(pos_or_key, int):
                    args_str += f"{args[pos_or_key]},"
                elif isinstance(pos_or_key, str):
                    arg_name = ""
                    if display_arg_name:
                        arg_name = f"{pos_or_key}="
                    args_str += f"{arg_name}{kwargs[pos_or_key]},"
                else:
                    raise ValueError(f"Only accepts int or str as positional or keyword arguments not {type(pos_or_key)}")
            
            print(f"{prefix}Finished {func.__name__}({args_str}) in {(end - start):.3f} seconds", flush=True)

            if result is not None:
                return result

        return wrapper

    return outer_dec

def tempo(starttime, endtime):
    """ Returns a string with the time elapsed between starttime,endtime """

    dif = round(endtime - starttime, 2)
    d = int(dif / 86400)
    leftseconds = dif - d * 86400
    h = int(leftseconds / 3600)
    m = int((leftseconds / 3600 - h) * 60)
    s = round(((leftseconds / 3600 - h) * 60 - m) * 60, 2)

    if d > 0:
        string = (
            " The program ran for "
            + str(d)
            + " days, "
            + str(h)
            + " h, "
            + str(m)
            + " m"
        )
    elif h > 0:
        string = (
            " The program ran for "
            + str(h)
            + " h, "
            + str(m)
            + " m, "
            + str(int(s))
            + " s"
        )
    elif m > 0:
        string = " The program ran for " + str(m) + " m, " + str(s) + " s"
    else:
        string = " The program ran for " + str(dif) + " s"

    return string

def inter_time(interval):
    """ Returns a string with the time interval."""

    d = int(interval / 86400)
    leftseconds = interval - d * 86400
    h = int(leftseconds / 3600)
    m = int((leftseconds / 3600 - h) * 60)
    s = round(((leftseconds / 3600 - h) * 60 - m) * 60, 2)

    if d > 0:
        string = (
            str(d)
            + " days, "
            + str(h)
            + " h, "
            + str(m)
            + " m"
        )
    elif h > 0:
        string = (
            str(h)
            + " h, "
            + str(m)
            + " m, "
            + str(int(s))
            + " s"
        )
    elif m > 0:
        string = str(m) + " m, " + str(s) + " s"
    else:
        string = str(round(interval, 2)) + " s"

    return string
