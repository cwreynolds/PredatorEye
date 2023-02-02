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

#    # Accumulated a new “training set” of the most recent N steps seen so far. (See
#    # https://cwreynolds.github.io/TexSyn/#20220421 and ...#20220424 for discussion
#    # of this parameter. Had been 1, then 100, then 200, then finally, infinity.)
#    # max_training_set_size = float('inf') # keep ALL steps in training set, use GPU.
#    max_training_set_size = 500 # Try smaller again, "yellow flowers" keeps failing.
#
#    # List of "pixel tensors".
#    fine_tune_images = []
#
#    # List of xy3 [[x,y],[x,y],[x,y]] for 3 prey centers.
#    fine_tune_labels = []
#
#    def update(pixel_tensor, prediction, prey_centers_xy3, step, directory):
#        # Assume the predator was "aiming for" that one but missed by a bit.
#        sorted_xy3 = df.sort_xy3_by_proximity_to_point(prey_centers_xy3, prediction)
#
#        # Accumulate the most recent "max_training_set_size" training samples.
#        global fine_tune_images
#        global fine_tune_labels
#
#        # Still collecting initial training set?
#        if len(fine_tune_images) < max_training_set_size:
#            # Still building dataset, append new samples to end.
#            fine_tune_images.append(pixel_tensor)
#            fine_tune_labels.append(sorted_xy3)
#        else:
#            # Once dataset has reached full size, insert sample at random index.
#            random_index = random.randrange(max_training_set_size)
#            fine_tune_images[random_index] = pixel_tensor
#            fine_tune_labels[random_index] = sorted_xy3
#            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            # TODO 20220922 verify
#            print('  random_index =', random_index)
#            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#        print('  fine_tune_images shape =', np.shape(fine_tune_images),
#              '-- fine_tune_labels shape =', np.shape(fine_tune_labels))

#    class FineTuningDataset:
#        """Collects and manages a dataset of training examples, which are tournament
#           images from previous simulation steps, labeled with the predators most
#            accurate prediction of prey location."""
#
#        # Instance constructor.
#        def __init__(self):
#            # List of "pixel tensors".
#            self.fine_tune_images = []
#
#            # List of xy3 for 3 prey centers, least aim error first.
#            # An xy3 is [[x,y],[x,y],[x,y]]
#            self.fine_tune_labels = []
#
#        # Accumulate a dataset of the most recent N steps seen so far.
#        max_dataset_size = 500
#
#
#    #    # Accumulated a new “training set” of the most recent N steps seen so far. (See
#    #    # https://cwreynolds.github.io/TexSyn/#20220421 and ...#20220424 for discussion
#    #    # of this parameter. Had been 1, then 100, then 200, then finally, infinity.)
#    #    # max_training_set_size = float('inf') # keep ALL steps in training set, use GPU.
#    #    max_training_set_size = 500 # Try smaller again, "yellow flowers" keeps failing.
#
#    #    # List of "pixel tensors".
#    #    fine_tune_images = []
#    #
#    #    # List of xy3 [[x,y],[x,y],[x,y]] for 3 prey centers.
#    #    fine_tune_labels = []
#
#    #    def update(pixel_tensor, prediction, prey_centers_xy3, step, directory):
#    #    def update(self, pixel_tensor, prediction, prey_centers_xy3, step, directory):
#        def update(self, pixel_tensor, prediction, prey_centers_xy3):
#            # Assume the predator was "aiming for" that one but missed by a bit.
#            sorted_xy3 = df.sort_xy3_by_proximity_to_point(prey_centers_xy3, prediction)
#
#    #        # Accumulate the most recent "max_training_set_size" training samples.
#    #        global fine_tune_images
#    #        global fine_tune_labels
#
#            # Still collecting initial training set?
#    #        if len(fine_tune_images) < max_training_set_size:
#            if len(self.fine_tune_images) < self.max_dataset_size:
#                # Still building dataset, append new samples to end.
#    #            fine_tune_images.append(pixel_tensor)
#    #            fine_tune_labels.append(sorted_xy3)
#                self.fine_tune_images.append(pixel_tensor)
#                self.fine_tune_labels.append(sorted_xy3)
#            else:
#                # Once dataset has reached full size, insert sample at random index.
#    #            random_index = random.randrange(max_training_set_size)
#    #            random_index = random.randrange(max_dataset_size)
#                random_index = random.randrange(self.max_dataset_size)
#    #            fine_tune_images[random_index] = pixel_tensor
#    #            fine_tune_labels[random_index] = sorted_xy3
#                self.fine_tune_images[random_index] = pixel_tensor
#                self.fine_tune_labels[random_index] = sorted_xy3
#    #            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#    #            # TODO 20220922 verify
#    #            print('  random_index =', random_index)
#    #            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#    #        print('  fine_tune_images shape =', np.shape(fine_tune_images),
#    #              '-- fine_tune_labels shape =', np.shape(fine_tune_labels))
#            print('  fine_tune_images shape =', np.shape(self.fine_tune_images),
#                  '-- fine_tune_labels shape =', np.shape(self.fine_tune_labels))

#    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#    # TODO 20230130 make FineTuningDataset into a class, one per Predator
#
#    class FineTuningDataset:
#        """Collects and manages a dataset of training examples, which are tournament
#           images from previous simulation steps, labeled with the predators most
#            accurate prediction of prey location."""
#
#        # Instance constructor.
#        def __init__(self):
#            # List of "pixel tensors".
#            self.fine_tune_images = []
#
#            # List of xy3 for 3 prey centers, least aim error first.
#            # An xy3 is [[x,y],[x,y],[x,y]]
#            self.fine_tune_labels = []
#
#        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        # TODO 20230201 at starvation time log predator's ftd size
#        # Accumulate a dataset of the most recent N steps seen so far.
#    #    max_dataset_size = 500
#        max_dataset_size = 150
#        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#        def update(self, pixel_tensor, prediction, prey_centers_xy3):
#            # Assume the predator was "aiming for" that one but missed by a bit.
#            sorted_xy3 = df.sort_xy3_by_proximity_to_point(prey_centers_xy3, prediction)
#
#            # Still collecting initial training set?
#            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            # TODO 20230201 at starvation time log predator's ftd size
#    #        if len(self.fine_tune_images) < self.max_dataset_size:
#            if self.size() < self.max_dataset_size:
#            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                # Still building dataset, append new samples to end.
#                self.fine_tune_images.append(pixel_tensor)
#                self.fine_tune_labels.append(sorted_xy3)
#            else:
#                # Once dataset has reached full size, insert sample at random index.
#                random_index = random.randrange(self.max_dataset_size)
#                self.fine_tune_images[random_index] = pixel_tensor
#                self.fine_tune_labels[random_index] = sorted_xy3
#
#            print('  fine_tune_images shape =', np.shape(self.fine_tune_images),
#                  '-- fine_tune_labels shape =', np.shape(self.fine_tune_labels))
#
#        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        # TODO 20230201 at starvation time log predator's ftd size
#        def size(self):
#            return len(self.fine_tune_images)
#        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#
#    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


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
