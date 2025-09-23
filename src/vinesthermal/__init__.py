"""The docstring for the library goes here!

We should explain generally what it does.

We should end with an example.
"""

###############################################################################
# Dependencies for the library.
import os, pandas
from os import fspath
from pathlib import Path

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage as ski
import sklearn as skl
#import torch


###############################################################################
# Utility Functions

def load_image(filepath):
    """Loads an image and returns the numpy image array."""
    im = ski.io.imread(flnm)
    im = np.array(im, dtype=float)
    return im / 255
def to_image(image):
    """If the given argument is an image, return it; otherwise, load the image
    as a file and return it."""
    try:
        return load_image(image)
    except Exception:
        image = np.array(image, dtype=float)
        if np.max(image) > 1:
            image = image / 255
        return image
def image_to_grayscale(im):
    """Returns a grayscale version of the given image."""
    return np.mean(im, axis=-1)

default_gabor_sizes = (4, 6, 8, 12, 16, 24)
default_gabor_angles = np.arange(8) / 8 * np.pi
def filter_image(image, gabor_sizes=None, gabor_angles=None):
    """Filters the image with a set of Gabor filters and returns the filtered
    images as an array.

    (We'll eventually need more documentation here.)
    """
    # Process arguements:
    if gabor_sizes is None:
        gabor_sizes = default_gabor_sizes
    if gabor_angles is None:
        gabor_angles = default_gabor_angles
    elif isinstance(gabor_angles, int):
        gabor_angles = np.arange(gabor_angles) / gabor_angles * np.pi
    # Code
    im = to_image(image)
    # Filter in many ways:
    output = []
    for gsz in gabor_sizes:
        sz_output = []
        for gang in gabor_angles:
            (real,imag) = ski.filters.gabor(im, theta=gang, frequency=1/gsz)
            sz_output.append(np.sqrt(real**2 + imag**2))
        sz_output = np.array(sz_output)
        output.append(sz_output)
    output = np.array(output)
    return(output)

def image_to_greenemph(im, sigma=10):
    """Returns a 1-channel version of the given RGB image
    that emphasizes the green channel."""
    green_image = im[:,:,1]
    redblue_conv_image = ski.filters.gaussian(
        (im[:,:,0] + im[:,:,2]) / 2,
        sigma=sigma,
        preserve_range=True)
    return green_image / (redblue_conv_image + 0.05)

