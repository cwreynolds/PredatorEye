#!/usr/bin/env python
# coding: utf-8
################################################################################
#
# FineTuningDataset.py
# FineTuningDataset class
# Manages the dataset of images and labels for fine-tuning Predator models.
# PredatorEye system
#
# 20220920 Split off from EvoCamoVsLearnPredPop.py (from ...ipynb)
# Copyright © 2022 Craig Reynolds. All rights reserved.
#
################################################################################

import numpy as np
import DiskFind as df

# Accumulated a new “training set” of the most recent N steps seen so far. (See
# https://cwreynolds.github.io/TexSyn/#20220421 and ...#20220424 for discussion
# of this parameter. Had been 1, then 100, then 200, then finally, infinity.)
# max_training_set_size = float('inf') # keep ALL steps in training set, use GPU.
max_training_set_size = 500 # Try smaller again, "yellow flowers" keeps failing.

# List of "pixel tensors".
fine_tune_images = []

# List of xy3 [[x,y],[x,y],[x,y]] for 3 prey centers.
fine_tune_labels = []

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TODO 20220921 the fine-tuning dataset had been a fixed length history of the
# last max_training_set_size simulation steps, holding both the input image and
# "best prediction".
#
# Now the dataset grows to size max_training_set_size as before, but thereafter
# each new training example is inserted at random, at a uniformly sampled index.
# This make the dataset represent a larger interval of simulated time at lower
# sampling density.

#def update(pixel_tensor, prediction, prey_centers_xy3, step, directory):
#    # Assume the predator was "aiming for" that one but missed by a bit.
#    sorted_xy3 = df.sort_xy3_by_proximity_to_point(prey_centers_xy3, prediction)
#
#    # Accumulate the most recent "max_training_set_size" training samples.
#    global fine_tune_images
#    global fine_tune_labels
#    fine_tune_images.append(pixel_tensor)
#    fine_tune_labels.append(sorted_xy3)
#
#    # If training set has become too large, slice off first element of each.
#    if len(fine_tune_images) > max_training_set_size:
#        fine_tune_images = fine_tune_images[1:]
#        fine_tune_labels = fine_tune_labels[1:]
#
#    print('  fine_tune_images shape =', np.shape(fine_tune_images),
#          '-- fine_tune_labels shape =', np.shape(fine_tune_labels))

import random

def update(pixel_tensor, prediction, prey_centers_xy3, step, directory):
    # Assume the predator was "aiming for" that one but missed by a bit.
    sorted_xy3 = df.sort_xy3_by_proximity_to_point(prey_centers_xy3, prediction)

    # Accumulate the most recent "max_training_set_size" training samples.
    global fine_tune_images
    global fine_tune_labels
    
    # Still collecting initial training set?
    if len(fine_tune_images) < max_training_set_size:
        # Still building dataset, append new samples to end.
        fine_tune_images.append(pixel_tensor)
        fine_tune_labels.append(sorted_xy3)
    else:
        # Once dataset has reached full size, insert sample at random index.
#        random_index = random.randint(0, max_training_set_size)
        random_index = random.randrange(max_training_set_size)
        fine_tune_images[random_index] = pixel_tensor
        fine_tune_labels[random_index] = sorted_xy3

    print('  fine_tune_images shape =', np.shape(fine_tune_images),
          '-- fine_tune_labels shape =', np.shape(fine_tune_labels))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
