#!/usr/bin/env python
# coding: utf-8
################################################################################
#
# FineTuningDataset.py -- PredatorEye system
#
# Manages the dataset of images and labels for fine-tuning Predator models.
#
# 20220920 Split off from EvoCamoVsLearnPredPop.py (from ...ipynb)
# Copyright © 2022 Craig Reynolds. All rights reserved.
#
################################################################################

import random
import numpy as np
import DiskFind as df

class FineTuningDataset:
    """Collects and manages a dataset of training examples, which are tournament
       images from previous simulation steps, labeled with the predators most
       accurate prediction of prey location."""

    # Instance constructor.
    def __init__(self):
        # List of "pixel tensors".
        self.fine_tune_images = []
        # List of xy3 for 3 prey centers, least aim error first.
        # An xy3 is [[x,y],[x,y],[x,y]]
        self.fine_tune_labels = []

    # Max size of dataset. First accumulate steps, then randomly replace.
    max_dataset_size = 150

    # Current size of dataset.
    def size(self):
        return len(self.fine_tune_images)

    def update(self, pixel_tensor, prediction, prey_centers_xy3):
        # Assume the predator was "aiming for" that one but missed by a bit.
        sorted_xy3 = df.sort_xy3_by_proximity_to_point(prey_centers_xy3, prediction)
        # Still collecting initial training set?
        if self.size() < self.max_dataset_size:
            # Still building dataset, append new samples to end.
            self.fine_tune_images.append(pixel_tensor)
            self.fine_tune_labels.append(sorted_xy3)
        else:
            # Once dataset has reached full size, insert sample at random index.
            random_index = random.randrange(self.max_dataset_size)
            self.fine_tune_images[random_index] = pixel_tensor
            self.fine_tune_labels[random_index] = sorted_xy3
        # Logging.
        print('  fine_tune_images shape =', np.shape(self.fine_tune_images),
              '-- fine_tune_labels shape =', np.shape(self.fine_tune_labels))
