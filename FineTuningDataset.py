#!/usr/bin/env python
# coding: utf-8
################################################################################
#
# FineTuningDataset.py
# FineTuningDataset class
# PredatorEye system
#
# 20220920 Split off from EvoCamoVsLearnPredPop.py (from ...ipynb)
# Copyright © 2022 Craig Reynolds. All rights reserved.
#
################################################################################

import numpy as np
import DiskFind as df

#    class FineTuningDataset:
#        """Manages the dataset of images and labels for fine-tuning."""
#
#        # Accumulated a new “training set” of the most recent N steps seen so far. (See
#        # https://cwreynolds.github.io/TexSyn/#20220421 and ...#20220424 for discussion
#        # of this parameter. Had been 1, then 100, then 200, then finally, infinity.)
#        # max_training_set_size = float('inf') # keep ALL steps in training set, use GPU.
#        max_training_set_size = 500 # Try smaller again, "yellow flowers" keeps failing.
#        # List of "pixel tensors".
#        fine_tune_images = []
#        # List of xy3 [[x,y],[x,y],[x,y]] for 3 prey centers.
#        fine_tune_labels = []
#
#        def update(self, pixel_tensor, prediction, step, directory):
#            # Assume the predator was "aiming for" that one but missed by a bit.
#            xy3 = read_3_centers_from_file(step, directory)
#            sorted_xy3 = df.sort_xy3_by_proximity_to_point(xy3, prediction)
#
#            # Accumulate the most recent "max_training_set_size" training samples.
#            self.fine_tune_images.append(pixel_tensor)
#            self.fine_tune_labels.append(sorted_xy3)
#
#            # If training set has become too large, slice off first element of each.
#            if len(self.fine_tune_images) > self.max_training_set_size:
#                self.fine_tune_images = self.fine_tune_images[1:]
#                self.fine_tune_labels = self.fine_tune_labels[1:]
#
#            print('  fine_tune_images shape =', np.shape(self.fine_tune_images),
#                  '-- fine_tune_labels shape =', np.shape(self.fine_tune_labels))
#
#
#    # Create a global FineTuningDataset object.
#    # (TODO globals are usually a bad idea, reconsider this.)
#    fine_tuning_dataset = FineTuningDataset()


# Following the pattern described in
# https://www.geeksforgeeks.org/singleton-pattern-in-python-a-complete-guide/
#    class SingletonClass(object):
#      def __new__(cls):
#        if not hasattr(cls, 'instance'):
#          cls.instance = super(SingletonClass, cls).__new__(cls)
#        return cls.instance


class FineTuningDataset:
    """Manages the dataset of images and labels for fine-tuning."""
    
    
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FineTuningDataset, cls).__new__(cls)
        return cls.instance

    
    

    # Accumulated a new “training set” of the most recent N steps seen so far. (See
    # https://cwreynolds.github.io/TexSyn/#20220421 and ...#20220424 for discussion
    # of this parameter. Had been 1, then 100, then 200, then finally, infinity.)
    # max_training_set_size = float('inf') # keep ALL steps in training set, use GPU.
    max_training_set_size = 500 # Try smaller again, "yellow flowers" keeps failing.
    # List of "pixel tensors".
    fine_tune_images = []
    # List of xy3 [[x,y],[x,y],[x,y]] for 3 prey centers.
    fine_tune_labels = []

#    def update(self, pixel_tensor, prediction, step, directory):
    def update(self, pixel_tensor, prediction, prey_centers_xy3, step, directory):
        # Assume the predator was "aiming for" that one but missed by a bit.
#        xy3 = read_3_centers_from_file(step, directory)
#        sorted_xy3 = df.sort_xy3_by_proximity_to_point(xy3, prediction)
        sorted_xy3 = df.sort_xy3_by_proximity_to_point(prey_centers_xy3, prediction)

        # Accumulate the most recent "max_training_set_size" training samples.
        self.fine_tune_images.append(pixel_tensor)
        self.fine_tune_labels.append(sorted_xy3)

        # If training set has become too large, slice off first element of each.
        if len(self.fine_tune_images) > self.max_training_set_size:
            self.fine_tune_images = self.fine_tune_images[1:]
            self.fine_tune_labels = self.fine_tune_labels[1:]

        print('  fine_tune_images shape =', np.shape(self.fine_tune_images),
              '-- fine_tune_labels shape =', np.shape(self.fine_tune_labels))


#    def update(pixel_tensor, prediction, step, directory):
#
#        self = FineTuningDataset.instance
#
#        # Assume the predator was "aiming for" that one but missed by a bit.
#        xy3 = read_3_centers_from_file(step, directory)
#        sorted_xy3 = df.sort_xy3_by_proximity_to_point(xy3, prediction)
#
#        # Accumulate the most recent "max_training_set_size" training samples.
#        self.fine_tune_images.append(pixel_tensor)
#        self.fine_tune_labels.append(sorted_xy3)
#
#        # If training set has become too large, slice off first element of each.
#        if len(self.fine_tune_images) > self.max_training_set_size:
#            self.fine_tune_images = self.fine_tune_images[1:]
#            self.fine_tune_labels = self.fine_tune_labels[1:]
#
#        print('  fine_tune_images shape =', np.shape(self.fine_tune_images),
#              '-- fine_tune_labels shape =', np.shape(self.fine_tune_labels))

