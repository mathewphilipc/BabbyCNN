from __future__ import print_function

import tensorflow as tf
import os

MINI_OR_FULL = "MINI"
MINI_DATASET_PATH = "/home/mathew/Desktop/NWPU-RESISC45-MINI"
FULL_DATASET_PATH = "/home/mathew/Desktop/NWPU-RESISC45"


MODE = 'folder'
DATASET_PATH = FULL_DATASET_PATH

MINI_N_CLASSES = 10
FULL_N_CLASSES = 45
N_CLASSES = FULL_N_CLASSES

print(N_CLASSES)

IMG_HEIGHT = 32 # original size = 256
IMG_WIDTH = 32 # original size = 256
CHANNELS = 3 # we have full-color images
