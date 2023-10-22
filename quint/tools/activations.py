import math


def rev_sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(0.5 * x))
