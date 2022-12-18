#!/usr/bin/env python
# coding: utf-8
################################################################################
#
# Tournament.py -- PredatorEye system
#
# Represents three Predators in a Tournament of the camouflage simulation.
#
# 20220924 Split off from EvoCamoVsLearnPredPop.py (from ...ipynb)
# Copyright © 2022 Craig Reynolds. All rights reserved.
#
################################################################################

import statistics
import DiskFind as df
import tensorflow as tf
from Predator import Predator

class Tournament:
    """Represents three Predators in a Tournament of the camouflage simulation."""
    
    class Member:
        """One Predator in a Tournament of the camouflage simulation."""
        def __init__(self, predator, tf_pixel_tensor, prey_centers_xy3):
            self.predator = predator
            self.tf_pixel_tensor = tf_pixel_tensor
            self.prey_centers_xy3 = prey_centers_xy3
            self.prediction = self.predator.model.predict(tf_pixel_tensor)[0]
            self.aim_error = df.aim_error(self.prediction, self.prey_centers_xy3)

    def __init__(self, pixel_tensor, prey_centers_xy3):
        # Store pixel data from current input image from TexSyn side.
        self.pixel_tensor = pixel_tensor
        # Also store it as a TF-style tensor
        # TODO 20220907 should eventually replace the non-TF data.
        #               now causes an error in Predator.fine_tune_model()
        self.tf_pixel_tensor = tf.convert_to_tensor([self.pixel_tensor])
        # Store the positions of all prey centers as xy on image.
        self.prey_centers_xy3 = prey_centers_xy3
        # Choose random tournament of three from population of Predators.
        # Build a list with each wrapped in Tournament Member objects.
        self.members = [self.Member(predator, self.tf_pixel_tensor, prey_centers_xy3)
                        for predator in Predator.choose_for_tournament(3)]
        # Rank members by aim_error (smallest first)
        self.members = sorted(self.members, key=lambda m: m.aim_error)
        # Sort predictions from the three Predators in a tournament, according to
        # ”accuracy” (least aim error).
        # TODO should this be computed on the fly in Tournament.ranked_predictions()?
        self.ranked_predictions_xy3 = [member.prediction for member in self.members]

    # Gets the list of 3 prey center predictions, ranked most accurate at front.
    def ranked_predictions(self):
        return self.ranked_predictions_xy3
    
    # Perform fine-tuning step on each Predator in this Tournament.
    def fine_tune_models(self):
        for member in self.members:
            member.predator.fine_tune_model(self.pixel_tensor,
                                            member.prediction,
                                            self.prey_centers_xy3)

    # Called at Tournament end to update Predator population if needed.
    def update_predator_population(self):
        worst_predator = self.members[-1].predator
        if worst_predator.starvation():
            global xxx_temp_starvation_count
            xxx_temp_starvation_count += 1
            print()
            print('starving!! ', xxx_temp_starvation_count, ', ',
                  worst_predator.step / xxx_temp_starvation_count, ', ',
                  "%.3f" % (xxx_temp_starvation_count / worst_predator.step),
                  sep='')
            # Replace worst predator in Tournament with offspring of other two.
            worst_predator.replace_in_population(self.members[0].predator,
                                                 self.members[1].predator)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # TODO 20221214 age tracking
            # TODO 20221216 print min and max in addition to mean.
            ages = []
            for p in Predator.population:
                ages.append(p.age())
#            print('average age:', statistics.mean(ages))
            print('age (min mean max):',
                  min(ages), statistics.mean(ages), max(ages))
            print('ages:', end = " ")
            for a in ages:
                print(a, end = " ")
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            print()


# TODO 20220913 temp ad hoc counter
xxx_temp_starvation_count = 0
