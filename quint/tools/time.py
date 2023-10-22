import logging
from functools import wraps
import time as t


def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = t.time()
        result = func(*args, **kwargs)
        end = t.time()
        logging.warning("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper
