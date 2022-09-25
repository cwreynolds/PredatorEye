#!/usr/bin/env python
# coding: utf-8
################################################################################
#
# EvoCamoVsLearnPredPop.py -- PredatorEye system
#
# Evolutionary Camouflage Versus a Learning Predator Population
# Top level for "predator side" of advarsarial camouflage evolution simulation.
#
# Copyright © 2022 Craig Reynolds. All rights reserved.
#
################################################################################


#    # **EvoCamoVsLearnPredPop.ipynb**
#    #
#    # August 23, 2022: this version runs “local mode” with both predator and prey running on the same machine.
#    #
#    # (The former behavior available with `Rube_Goldberg_mode = True`)
#
#    # In[1]:
#
#
#    # "Rube Goldberg" mode refers to running camouflage evolution on my laptop while
#    # running predator vision in cloud via Colab. State is passed back and forth via
#    # files on Google Drive.
#
#    # TODO 20220822
#    # Rube_Goldberg_mode = True
#    Rube_Goldberg_mode = False
#
#    def if_RG_mode(for_RG_mode, for_normal_mode):
#        return for_RG_mode if Rube_Goldberg_mode else for_normal_mode
#
#    # PredatorEye directory on Drive.
#    pe_directory = '/content/drive/My Drive/PredatorEye/'
#
#    # Shared "communication" directory on Drive.
#    shared_directory = if_RG_mode(pe_directory + 'evo_camo_vs_static_fcd/',
#                                  '/Users/cwr/camo_data/comms/')
#
#    # This was meant (20220716) to allow reading original pre-trained model from
#    # Google Drive, but I'll need to retrain it for M1 (Apple Silicon).
#    g_drive_pe_dir = ('/Users/cwr/Library/CloudStorage/' +
#                      'GoogleDrive-craig.w.reynolds@gmail.com/' +
#                      'My Drive/PredatorEye/')
#
#    # Directory for pre-trained Keras/TensorFlow models.
#    saved_model_directory = if_RG_mode(pe_directory, g_drive_pe_dir) + 'saved_models/'
#
#
#    print('Rube_Goldberg_mode =', Rube_Goldberg_mode)
#    print('shared_directory =', shared_directory)
#    print('saved_model_directory =', saved_model_directory)
#
#    # Pathname of pre-trained Keras/TensorFlow model
#    trained_model = saved_model_directory + '20220321_1711_FCD6_rc4'
#
#    # Directory on Drive for storing fine-tuning dataset.
#    fine_tuning_directory = shared_directory + 'fine_tuning/'
#
#    my_prefix = "find_"
#    other_prefix = "camo_"
#
#    my_suffix =  ".txt"
#    # other_suffix = ".jpeg"
#    other_suffix = ".png"

fcd_image_size = 1024
fcd_disk_size = 201

import time
import PIL
from pathlib import Path

from tensorflow import keras
import numpy as np
import random
import math

import tensorflow as tf
print('TensorFlow version:', tf.__version__)

from tensorflow.keras import backend as keras_backend
keras_backend.set_image_data_format('channels_last')

#    # Import DiskFind utilities for PredatorEye.
#    import sys
#    if Rube_Goldberg_mode:
#        sys.path.append('/content/drive/My Drive/PredatorEye/shared_code/')
#    else:
#        sys.path.append('/Users/cwr/Documents/code/PredatorEye/')





import DiskFind as df

from Predator import Predator
import FineTuningDataset as ftd
from Tournament import Tournament
import PredatorServer as ps

#ps.shared_directory = shared_directory
ps.shared_directory = '/Users/cwr/camo_data/comms/'


#    shared_directory = if_RG_mode(pe_directory + 'evo_camo_vs_static_fcd/',
#                                  '/Users/cwr/camo_data/comms/')

# Directory for pre-trained Keras/TensorFlow models.
#saved_model_directory = if_RG_mode(pe_directory, g_drive_pe_dir) + 'saved_models/'
#g_drive_pe_dir = ('/Users/cwr/Library/CloudStorage/' +
#                  'GoogleDrive-craig.w.reynolds@gmail.com/' +
#                  'My Drive/PredatorEye/')
#saved_model_directory = g_drive_pe_dir + 'saved_models/'
g_drive_pe_dir = ('/Users/cwr/Library/CloudStorage/' +
                  'GoogleDrive-craig.w.reynolds@gmail.com/' +
                  'My Drive/PredatorEye/')
saved_model_directory = g_drive_pe_dir + 'saved_models/'

# Pathname of pre-trained Keras/TensorFlow model
trained_model = saved_model_directory + '20220321_1711_FCD6_rc4'




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

# Calculates RELATIVE disk radius on the fly -- rewrite later.
def fcd_disk_radius():
    return (float(fcd_disk_size) / float(fcd_image_size)) / 2

# Given two tensors of 2d point coordinates, return a tensor of the Cartesian
# distance between corresponding points in the input tensors.
def corresponding_distances(y_true, y_pred):
    true_pos_x, true_pos_y = tf.split(y_true, num_or_size_splits=2, axis=1)
    pred_pos_x, pred_pos_y = tf.split(y_pred, num_or_size_splits=2, axis=1)
    dx = true_pos_x - pred_pos_x
    dy = true_pos_y - pred_pos_y
    distances = tf.sqrt(tf.square(dx) + tf.square(dy))
    return distances

# 20211231 copied from Find_Concpocuous_Disk
def in_disk(y_true, y_pred):
    distances = corresponding_distances(y_true, y_pred)
    # relative_disk_radius = (float(fcd_disk_size) / float(fcd_image_size)) / 2

    # From https://stackoverflow.com/a/42450565/1991373
    # Boolean tensor marking where distances are less than relative_disk_radius.
    # insides = tf.less(distances, relative_disk_radius)
    insides = tf.less(distances, fcd_disk_radius())
    map_to_zero_or_one = tf.cast(insides, tf.int32)
    return map_to_zero_or_one

dependencies = { 'in_disk': in_disk }

def read_default_pre_trained_model():
    print('Reading pre-trained model from:', trained_model)
    return keras.models.load_model(trained_model, custom_objects=dependencies)


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
