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

import random
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
        random_index = random.randrange(max_training_set_size)
        fine_tune_images[random_index] = pixel_tensor
        fine_tune_labels[random_index] = sorted_xy3
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # TODO 20220922 verify
        print('  random_index =', random_index)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    print('  fine_tune_images shape =', np.shape(fine_tune_images),
          '-- fine_tune_labels shape =', np.shape(fine_tune_labels))
