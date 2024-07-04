import numpy as np


def arg_closest(a, target):
    return np.argmin(np.abs(a - target))


def linear(x, x0, y0, x1, y1):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def to_cartesian(theta, radius):
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


def to_polar(x, y):
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return theta, radius


def to_degrees(theta):
    return theta * 180. / np.pi


def to_radians(theta):
    return theta * np.pi / 180.
