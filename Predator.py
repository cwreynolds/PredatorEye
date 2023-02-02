#!/usr/bin/env python
# coding: utf-8
################################################################################
#
# Predator.py -- PredatorEye system
#
# Predator class
#
# 20220919 Split off from EvoCamoVsLearnPredPop.py (from ...ipynb)
# Copyright © 2022 Craig Reynolds. All rights reserved.
#
################################################################################

import math
import random
import numpy as np
import DiskFind as df
import tensorflow as tf
from FineTuningDataset import FineTuningDataset

class Predator:
    """Represents a Predator in the camouflage simulation. It has a CNN-based
       model of visual hunting that identifies the position of likely prey."""

    # Global list of active Predators (as a class variable).
    population = []

    # Cache the standard default_pre_trained_model (as a class variable).
    default_pre_trained_model = None
    
    # Class parameter: default strength of noise used to jiggle CNN models.
    jiggle_strength = 0.003
    
    # How much recent predation success data is kept:
    success_history_max_length = 20
    # And how much of it must be non-zero to avoid starvation.
    success_history_ratio = 0.4
    success_history_min_meals = success_history_max_length*success_history_ratio

    # Keep track of how often selected prey is nearest center:
    nearest_center = 0
    
    # TODO 20220911 this is a goofy hack to avoid passing the "step" parameter
    # to Predator.fine_tune_model() via Tournament.fine_tune_models() for the
    # sole purpose of logging for "nearest_center" which really needs redesign.
    step = 0
    
    # Only used for generating unique name.
    instance_counter = 0
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO 20230130 make FineTuningDataset into a class, one per Predator
    #               change this to be a absolute number of training examples
    min_ftd_size = 25
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Instance constructor.
    def __init__(self):
        # Each Predator has its own copy of a prey-finding CNN-based model.
        self.model = []
        # By default add this new Predator to the population (TODO reconsider)
        Predator.population.append(self)
        # Keep history of predation events: was hunt a success or a failure?
        self.successes = []
        self.birthday = self.step
        self.name = 'predator_' + str(Predator.instance_counter)
        Predator.instance_counter += 1
        self.previous_in_disk = 0
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # TODO 20230130 make FineTuningDataset into a class, one per Predator
        self.ftd = FineTuningDataset()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # number of steps since this Predator was created
    def age(self):
        return self.step - self.birthday

    # Apply fine-tuning to (originally pre-trained) predator model. Use recent
    # steps as  training set. Assume they were "near misses" and so training
    # label is actual (ground truth) center of disk nearest prediction.
    def fine_tune_model(self, pixel_tensor, prediction, prey_centers_xy3, report):
        self.log_center_preference(prediction, prey_centers_xy3, report)

        # Convert training data list to np arrays
        # (TODO 20230107 I think these are identical for all three Predators in
        #    a Tournament. Could be precomputed in Tournament.fine_tune_models()
        #    and passed in here.)
#        images_array = np.array(ftd.fine_tune_images)
#        labels_array = np.array([x[0] for x in ftd.fine_tune_labels])
        images_array = np.array(self.ftd.fine_tune_images)
        labels_array = np.array([x[0] for x in self.ftd.fine_tune_labels])

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # TODO 20230130 make FineTuningDataset into a class, one per Predator
#        # Skip fine-tuning until dataset is large enough (10% of max size).
##        if images_array.shape[0] > (ftd.max_training_set_size * 0.1):
#        if images_array.shape[0] > (self.ftd.max_dataset_size * 0.1):
#        # Skip fine-tuning until dataset is large enough (25 = 5% of max size).
#        if images_array.shape[0] > (self.ftd.max_dataset_size * 0.05):

        # Skip fine-tuning until dataset is large enough.
        if images_array.shape[0] > self.min_ftd_size:
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # TODO 20220823 -- run fine-tuning on CPU only.
#            print('Running on CPU ONLY!')
            with tf.device('/cpu:0'):
                # Do fine-tuning training step using data accumulated during run.
                history = self.model.fit(x=images_array, y=labels_array, verbose=0)
            # For logging: record in_disk value from this fine-tune
            self.previous_in_disk = history.history['in_disk'][-1]
            h = '  ' + self.name + ', in_disk:'
            print(h, '{:0.2f}'.format(self.previous_in_disk))
        
        # Keep recent win/loss record for this predator for starvation pruning.
        self.record_predation_success(prediction, prey_centers_xy3)

    def log_center_preference(self, prediction, prey_centers_xy3, report):
        # Assume the predator was "aiming for" that one but missed by a bit.
        sorted_xy3 = df.sort_xy3_by_proximity_to_point(prey_centers_xy3, prediction)

        # TODO temp: keep track of how often selected prey is nearest center:
        # TODO 20230109 needed?
        #      (much later 20220911, no I don't think so)
        #      (even later 20230109, corresponds to sort() vs. sorted())
        temp = prey_centers_xy3.copy()
        sorted_by_dist_to_center = df.sort_xy3_by_proximity_to_point(temp, [0.5, 0.5])
        if sorted_by_dist_to_center[0] == sorted_xy3[0]:
            Predator.nearest_center += 1
        if report:
            nc = self.nearest_center / 3  # ad hoc adjustment to ad hoc metric
            print('  nearest_center:',
                  str(int(100 * float(nc) / (self.step + 1))) + '%',
                  '(nearest_center =', str(int(nc)) + ',',
                  'steps =', str(self.step + 1) + ')')

    # Copy the neural net model of a given predator into this one.
    def copy_model_of_another_predator(self, another_predator):
        self.copy_model(another_predator.model)

    # Copy a given neural net model into this one predator. (From "Make
    # deep copy of keras model" https://stackoverflow.com/a/54368176/1991373)
    def copy_model(self, other_model):
        # Clone layer structure of other model.
        self.model = tf.keras.models.clone_model(other_model)
        # Compile newly cloned model.
        df.compile_disk_finder_model(self.model)
        # Copy weights of other model.
        self.model.set_weights(other_model.get_weights())

    # Modify this Predator's model by adding signed noise to its weights.
    def jiggle_model(self, strength = None):
        if strength == None:
            strength = Predator.jiggle_strength
        weight_perturbation(self.model, tf.constant(strength))

    # Print the "middle" weight of each layer of this Predator's Keras model.
    def print_model_trace(self):
        for layer in self.model.layers:
            trainable_weights = layer.trainable_variables
            for weight in trainable_weights:
                weight_shape = tf.shape(weight)
                total_size = tf.math.reduce_prod(weight_shape)
                reshape_1d = tf.reshape(weight, [total_size])
                # Take "middle" parameter of layer.
                middle = math.floor(total_size / 2)
                value = reshape_1d[middle].numpy()
                print(round(value, 2), end = " ")
        print()

    # Create the given number of Predator objects
    # (TODO maybe the pretrained model should be an arg?)
    def initialize_predator_population(population_size, pre_trained_model):
    
        Predator.default_pre_trained_model = pre_trained_model
    
        for i in range(population_size):
            p = Predator()
            print('Predator instance address:', id(p))
            # TODO 20220907 maybe just do this by default in constructor?
            p.copy_model(Predator.default_pre_trained_model)
            p.jiggle_model()
            p.print_model_trace()
        print('Created population of', population_size, 'Predators.')

    # Randomly select "size" unique Predators from population for a Tournament.
    def choose_for_tournament(size = 3):
        assert len(Predator.population) >= size, "Population smaller than tournament size."
        return random.sample(Predator.population, size)
    
    # Maintain record of recent hunts and which ones were successfu.
    def record_predation_success(self, prediction_xy, prey_centers_xy3):
        radius = df.relative_disk_radius()
        distance = df.aim_error(prediction_xy, prey_centers_xy3)
        # Append a 0 (fail) or 1 (succeed) to history.
        self.successes.append(0 if distance < radius else 1)
        # Trim to max length.
        self.successes = self.successes[-self.success_history_max_length:]

    # Defines starvation as succeeding less than fraction of preceding hunts.
    def starvation(self):
        starving = False
        if len(self.successes) == self.success_history_max_length:
            count = sum(self.successes)
            if count < self.success_history_min_meals:
                starving = True
        return starving
    
    # When a Predator starves, replace it with an "offspring" of two others.
    # TODO currently, random choice between:
    #      1: randomly choose one parent's model, copy, and jiggle.
    #      2: copy default_pre_trained_model and jiggle.
    def replace_in_population(self, parent_a, parent_b):
        if random.choice([True, False]):
            parent = random.choice([parent_a, parent_b])
            self.copy_model_of_another_predator(parent)
        else:
            self.copy_model(Predator.default_pre_trained_model)
        self.jiggle_model(0.5 * Predator.jiggle_strength)
        self.successes = []
        self.birthday = self.step
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # TODO 20230130 make FineTuningDataset into a class, one per Predator
        self.ftd = FineTuningDataset()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        print('reinitializing predator', id(self))

# Utility based on https://stackoverflow.com/a/64542651/1991373
# TODO 20220907 added this to avoid always getting the same random_weights
# @tf.function

@tf.function(reduce_retracing=True)
def weight_perturbation(model, max_range):
    """Add noise to all weights in a Keras model."""
    for layer in model.layers:
        trainable_weights = layer.trainable_variables
        for weight in trainable_weights:
            random_weights = tf.random.uniform(tf.shape(weight),
                                               max_range / -2,
                                               max_range / 2,
                                               dtype=tf.float32)
            weight.assign_add(random_weights)
