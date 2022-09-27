#!/usr/bin/env python
# coding: utf-8
################################################################################
#
# EvoCamoVsLearnPredPop.py -- PredatorEye system
#
# Evolutionary Camouflage Versus a Learning Predator Population
#
# Top level for "predator side" of adversarial camouflage evolution simulation.
# Used in conjunction with evolutionary camouflage texture synthesis running in
# a separate process, communicating through files in a shared "comms" directory.
# This file is just the glue to join these components together and specify a
# handful of configuration parameters.
#
# Copyright © 2022 Craig Reynolds. All rights reserved.
#
################################################################################

import DiskFind as df
import tensorflow as tf
import PredatorServer as ps
import FineTuningDataset as ftd

from tensorflow import keras
from Predator import Predator
from Tournament import Tournament

print('TensorFlow version:', tf.__version__)

# TODO don't know if I need this to make format explicit, keeping in case I do.
from tensorflow.keras import backend as keras_backend
keras_backend.set_image_data_format('channels_last')

# Define absolute pathnames on local file system. These are for Craig's laptop.
#
# Set absolute local pathname of shared "comms" directory.
ps.shared_directory = '/Users/cwr/camo_data/comms/'
# Project directory on Google Drive, mounted on local file system.
g_drive_pe_dir = ('/Users/cwr/Library/CloudStorage/' +
                  'GoogleDrive-craig.w.reynolds@gmail.com/' +
                  'My Drive/PredatorEye/')
# Directory of pre-trained Keras/TensorFlow models.
saved_model_directory = g_drive_pe_dir + 'saved_models/'
# Pathname of pre-trained Keras/TensorFlow model
trained_model = saved_model_directory + '20220321_1711_FCD6_rc4'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calculates RELATIVE disk radius on the fly -- rewrite later.
#    fcd_image_size = 1024
#    fcd_disk_size = 201
#    #def fcd_disk_radius():
#    #    return (float(fcd_disk_size) / float(fcd_image_size)) / 2
#
#    def fcd_disk_radius_old():
#        return (float(fcd_disk_size) / float(fcd_image_size)) / 2
#
#    def fcd_disk_radius():
#        o = fcd_disk_radius_old()
#        n = df.relative_disk_radius()
#        assert o != n, "Radius mismatch."
#        return n
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # Read pre-trained model
# 
# As I integrate this into the Predator class, this is no longer “Read pre-trained model” but more like “Some utilities for reading the pre-trained model”

# In[3]:


# Read pre-trained TensorFlow "predator vision" model.

# print('Reading pre-trained model from:', trained_model)

# ad hoc workaround suggested on https://stackoverflow.com/q/66408995/1991373
#
# dependencies = {
#     'hamming_loss': tfa.metrics.HammingLoss(mode="multilabel", name="hamming_loss"),
#     'attention': attention(return_sequences=True)
# }
#
# dependencies = {
#     'valid_accuracy': ValidAccuracy
# }


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#    # Given two tensors of 2d point coordinates, return a tensor of the Cartesian
#    # distance between corresponding points in the input tensors.
#    def corresponding_distances(y_true, y_pred):
#        true_pos_x, true_pos_y = tf.split(y_true, num_or_size_splits=2, axis=1)
#        pred_pos_x, pred_pos_y = tf.split(y_pred, num_or_size_splits=2, axis=1)
#        dx = true_pos_x - pred_pos_x
#        dy = true_pos_y - pred_pos_y
#        distances = tf.sqrt(tf.square(dx) + tf.square(dy))
#        return distances
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TODO 20220927 isn't this now defined in DiskFind? Can't we use that one?
#               Merge them carefully.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 20211231 copied from Find_Concpocuous_Disk
#def in_disk(y_true, y_pred):
#    distances = corresponding_distances(y_true, y_pred)
#    # relative_disk_radius = (float(fcd_disk_size) / float(fcd_image_size)) / 2
#
#    # From https://stackoverflow.com/a/42450565/1991373
#    # Boolean tensor marking where distances are less than relative_disk_radius.
#    # insides = tf.less(distances, relative_disk_radius)
#    insides = tf.less(distances, fcd_disk_radius())
#    map_to_zero_or_one = tf.cast(insides, tf.int32)
#    return map_to_zero_or_one
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# dependencies = { 'in_disk': in_disk }
#dependencies = { 'in_disk': df.in_disk }
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#def read_default_pre_trained_model():
#    print('Reading pre-trained model from:', trained_model)
#    return keras.models.load_model(trained_model, custom_objects=dependencies)

def read_default_pre_trained_model():
    print('Reading pre-trained model from:', trained_model)
    return keras.models.load_model(trained_model,
                                   custom_objects={ 'in_disk': df.in_disk })
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Create population of Predators.
Predator.initialize_predator_population(20, read_default_pre_trained_model())
print('population size:', len(Predator.population))




# RUN SIMULATION
#
# Keep track of how often selected prey is nearest center:
Predator.nearest_center = 0
#
# Flush out obsolete files in comms directory.
ps.clean_up_communication_directory()
#
# Start fresh run defaulting to step 0.
ps.start_run()
