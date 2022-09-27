#!/usr/bin/env python
# coding: utf-8
################################################################################
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

# Set absolute local pathname of shared "comms" directory (for Craig's laptop).
ps.shared_directory = '/Users/cwr/camo_data/comms/'

# Read pre-trained "find conspicuous disk" model as Keras/TensorFlow neural net.
def read_default_pre_trained_model():
    # Project directory on Google Drive, mounted on local file system.
    g_drive_pe_dir = ('/Users/cwr/Library/CloudStorage/' +
                      'GoogleDrive-craig.w.reynolds@gmail.com/' +
                      'My Drive/PredatorEye/')
    # Directory of pre-trained Keras/TensorFlow models.
    saved_model_directory = g_drive_pe_dir + 'saved_models/'
    # Pathname of pre-trained Keras/TensorFlow model
    trained_model = saved_model_directory + '20220321_1711_FCD6_rc4'
    print('Reading pre-trained model from:', trained_model)
    return keras.models.load_model(trained_model,
                                   custom_objects={ 'in_disk': df.in_disk })

# Create population of Predators.
Predator.initialize_predator_population(20, read_default_pre_trained_model())
print('population size:', len(Predator.population))

# RUN SIMULATION
#
# Keep track of how often selected prey is nearest center:
Predator.nearest_center = 0
#
# Flush out any left-over files from previous run in comms directory.
ps.clean_up_communication_directory()
#
# Start fresh run.
ps.start_run()
