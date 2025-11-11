# interpolate.py
import numpy as np

def interpolate(z1, z2, steps=10):
    return [z1 * (1 - a) + z2 * a for a in np.linspace(0, 1, steps)]

def edit_attribute(z, direction, strength=2.0):
    return z + direction * strength