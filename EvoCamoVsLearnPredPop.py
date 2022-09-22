#!/usr/bin/env python
# coding: utf-8

# # Evolutionary Camouflage Versus a Learning Predator Population
# 
# ---
# 
# **EvoCamoVsLearnPredPop.ipynb**
# 
# August 23, 2022: this version runs “local mode” with both predator and prey running on the same machine.
# 
# (The former behavior available with `Rube_Goldberg_mode = True`)

# In[1]:


# "Rube Goldberg" mode refers to running camouflage evolution on my laptop while
# running predator vision in cloud via Colab. State is passed back and forth via
# files on Google Drive.

# TODO 20220822
# Rube_Goldberg_mode = True
Rube_Goldberg_mode = False

def if_RG_mode(for_RG_mode, for_normal_mode):
    return for_RG_mode if Rube_Goldberg_mode else for_normal_mode

# PredatorEye directory on Drive.
pe_directory = '/content/drive/My Drive/PredatorEye/'

# Shared "communication" directory on Drive.
shared_directory = if_RG_mode(pe_directory + 'evo_camo_vs_static_fcd/',
                              '/Users/cwr/camo_data/comms/')

# This was meant (20220716) to allow reading original pre-trained model from
# Google Drive, but I'll need to retrain it for M1 (Apple Silicon).
g_drive_pe_dir = ('/Users/cwr/Library/CloudStorage/' +
                  'GoogleDrive-craig.w.reynolds@gmail.com/' +
                  'My Drive/PredatorEye/')

# Directory for pre-trained Keras/TensorFlow models.
saved_model_directory = if_RG_mode(pe_directory, g_drive_pe_dir) + 'saved_models/'


print('Rube_Goldberg_mode =', Rube_Goldberg_mode)
print('shared_directory =', shared_directory)
print('saved_model_directory =', saved_model_directory)

# Pathname of pre-trained Keras/TensorFlow model
trained_model = saved_model_directory + '20220321_1711_FCD6_rc4'

# Directory on Drive for storing fine-tuning dataset.
fine_tuning_directory = shared_directory + 'fine_tuning/'

my_prefix = "find_"
other_prefix = "camo_"

my_suffix =  ".txt"
# other_suffix = ".jpeg"
other_suffix = ".png"

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

# Import DiskFind utilities for PredatorEye.
import sys
if Rube_Goldberg_mode:
    sys.path.append('/content/drive/My Drive/PredatorEye/shared_code/')
else:
    sys.path.append('/Users/cwr/Documents/code/PredatorEye/')
import DiskFind as df

from Predator import Predator
################################################################################
# TODO 20220921 re-refactor FineTuningDataset - why bother with a class?
#from FineTuningDataset import FineTuningDataset
import FineTuningDataset as ftd
################################################################################


# # Ad hoc “predator server”

# In[2]:


# Top level: wait for camo_xxx.jpeg files to appear, respond with find_xxx.txt
def start_run(step = 0):
    # Occurred to me on 20220915 that this was missing. Does it matter?
    df.reset_random_seeds(20220915)
    if step == 0:
        print('Start run in', shared_directory )
    else:
        print('Continue run at step', step, ' in', shared_directory)
    while True:
        perform_step(step, shared_directory)
        step += 1

# Continue from from the last camo_xxx.jpeg file.
def restart_run():
    start_run(newest_file_from_other(shared_directory))

# Single step: wait for camo file, write response, delete previous response.
def perform_step(step, directory):
    ############################################################################
    # TODO 20220909 add verbose "start of step" logging -- getting hard to read.
    print()
    print('step', step)
    print()
    ############################################################################
    wait_for_reply(step, shared_directory)
    write_response_file(step, shared_directory)
    delete_find_file(step - 1, shared_directory)

# Read image file for step, apply pre-trained model, write response file.
def write_response_file(step, directory):
    # Read image file and check for expected format.
    image_pathname = make_camo_pathname(step, directory)
    pixel_tensor = df.read_image_file_as_pixel_tensor(image_pathname)
    assert df.check_pixel_tensor(pixel_tensor), ('wrong file format: ' +
                                                 image_pathname)
    # TODO 20220907 can/should this TF version replace pixel_tensor below?
    #               no this causes an error in Predator.fine_tune_model()
    tf_pixel_tensor = tf.convert_to_tensor([pixel_tensor])
    
    # Collect self-supervision signal from TexSyn: positions of prey centers.
    prey_centers_xy3 = read_3_centers_from_file(step, directory)
    # Create a Tournament with 3 randomly selected Predators from population.
    tournament = Tournament(pixel_tensor, prey_centers_xy3)
    
    # Write response file (contains 3 xy positions as 6 floats on [0,1]).
    response_string = df.xy3_to_str(tournament.ranked_predictions())
    verify_comms_directory_reachable()
    with open(make_find_pathname(step, directory), 'w') as file:
        file.write(response_string)
    print('Wrote ' + "'" + response_string + "'",
          'to response file', Path(make_find_pathname(step, directory)).name)
    
    # Merge this step's image into fine-tuning dataset, and related bookkeeping.
    best_prediction = tournament.ranked_predictions()[0]
    ftd.update(pixel_tensor, best_prediction, prey_centers_xy3, step, directory)

    # Update the Predator population in case of starvation.
    tournament.update_predator_population()

    # Fine-tune models of each Predator in Tournament.
    Predator.step = step  # TODO 20220911 Goofy hack.
    tournament.fine_tune_models()

# Delete the given file, presumably after having written the next one.
def delete_find_file(step, directory):
    # Why doesn't pathlib provide a Path.remove() method like os?
    # TODO oh, missing_ok was added at pathlib version 3.8.
    # Path(makeMyPathname(step, directory)).unlink(missing_ok=True)
    p = Path(make_find_pathname(step, directory))
    if p.exists():
        p.unlink()

# Delete any remaining file in commuications directory to start a new run.
def clean_up_communication_directory():
    def delete_directory_contents(directory_path):
        for path in directory_path.iterdir():
            print('Removing from communication directory:', path)
            if path.is_dir():
                delete_directory_contents(path)
                path.rmdir()
            else:
                path.unlink()
    delete_directory_contents(Path(shared_directory))

# From pathname for file of given step number from the "other" agent.
def make_camo_pathname(step, directory):
    return directory + other_prefix + str(step) + other_suffix

# Form pathname for "find_xx.txt" response file from "this" agent.
def make_find_pathname(step, directory):
    return directory + my_prefix + str(step) + my_suffix

# Form pathname for "prey_xx.txt" ground truth file from "other" agent.
def make_prey_pathname(step, directory):
    return directory + 'prey_' + str(step) + '.txt'

# Used to ping the comms directory when it seems hung.
def write_ping_file(count, step, directory):
    pn = directory + 'ping_cloud_' + str(step) + '.txt'
    verify_comms_directory_reachable()
    with open(pn, 'w') as file:
        file.write(str(count))
    print('Ping comms: ', count, pn)

# Wait until other agent's file for given step appears.
def wait_for_reply(step, directory):
    camo_pathname = Path(make_camo_pathname(step, directory))
    camo_filename = camo_pathname.name
    prey_pathname = Path(make_prey_pathname(step, directory))
    prey_filename = prey_pathname.name
    print('Waiting for', camo_filename, 'and', prey_filename, '...',
          end='', flush=True)
    start_time = time.time()
    # Loop until both files are present, waiting 1 second between tests.
    test_count = 0
    while not (is_file_present(camo_pathname) and
               is_file_present(prey_pathname)):
        time.sleep(1)
        test_count += 1
        if (test_count % 100) == 0:
            write_ping_file(test_count, step, directory)
    print(' done, elapsed time:', int(time.time() - start_time), 'seconds.')

# Like fs::exists()
def is_file_present(file):
    result = False
    verify_comms_directory_reachable()
    filename = Path(file).name
    directory = Path(file).parent
    for i in directory.iterdir():
        if i.name == filename:
            result = True
    return result

# Returns the step number of the newest file from "other" in given directory.
# (So if "camo_573.jpeg" is the only "other" file there, returns int 573)
def newest_file_from_other(directory):
    steps = [0]  # Default to zero in case dir is empty.
    for filename in Path(directory).iterdir():
        name = filename.name
        if other_prefix == name[0:len(other_prefix)]:
            steps.append(int(name.split(".")[0].split("_")[1]))
    return max(steps)

# Read ground truth prey center location data provided in "prey_n.txt" file.
def read_3_centers_from_file(step, directory):
    # Read contents of file as string.
    verify_comms_directory_reachable()
    with open(make_prey_pathname(step, directory), 'r') as file:
        prey_centers_string = file.read()
    # Split string at whitespace, map to 6 floats, reshape into 3 xy pairs.
    # (TODO could probably be rewritten cleaner with "list comprehension")
    array = np.reshape(list(map(float, prey_centers_string.split())), (3, 2))
    return array.tolist()

# Keep log of in_disk metric.
def write_in_disk_log(step, history):
    if step % 10 == 0:
        in_disk = history.history["in_disk"][0]
        pathname = shared_directory + 'in_disk_log.csv'
        verify_comms_directory_reachable()
        with open(pathname, 'a') as file:
            if step == 0:
                file.write('step,in_disk\n')
            file.write(str(step) + ',' + "{:.4f}".format(in_disk) + '\n')

# Just wait in retry loop if shared "comms" directory become unreachable.
# Probably will return shortly, better to wait than signal a file error.
# (This is called from places with a local "directory" but it uses global value.)
def verify_comms_directory_reachable():
    seconds = 0
    # shared_directory_pathname = Path(shared_directory)
    # while not shared_directory_pathname.is_dir():
    while not Path(shared_directory).is_dir():
        print("Shared “comms” directory,", shared_directory, 
              "has been inaccessible for", seconds, "seconds.")
        time.sleep(1)  # wait 1 sec
        seconds += 1

################################################################################
# TODO 20220919 moved these to DiskFind.py
#
#    # Given 3 prey positions ("xy3"), sort them by proximity to "point" (prediction)
#    def sort_xy3_by_proximity_to_point(xy3, point):
#        # print('xy3 =', xy3)
#        xy3_plus_distance = [[df.dist2d(xy, point), xy] for xy in xy3]
#        # print('xy3_plus_distance =', xy3_plus_distance)
#        sorted_xy3_plus_key = sorted(xy3_plus_distance, key=lambda x: x[0])
#        # print('sorted_xy3_plus_key =', sorted_xy3_plus_key)
#        sorted_xy3 = [x[1] for x in sorted_xy3_plus_key]
#        # print('sorted_xy3 =', sorted_xy3)
#        return sorted_xy3
#
#    # Convert xy3 to string: [[x,y], [p,q], [r,s]] -> 'x y p q r s '
#    def xy3_to_str(xy3):
#        return ''.join('%s ' % i for i in flatten_nested_list(xy3))
#
#    def flatten_nested_list(nested_list):
#        return [item for sublist in nested_list for item in sublist]
################################################################################


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


# # Tournament class

# In[7]:


class Tournament:
    """Represents three Predators in a Tournament of the camouflage simulation."""
    
    class Member:
        """One Predator in a Tournament of the camouflage simulation."""
        def __init__(self, predator, tf_pixel_tensor, prey_centers_xy3):
            self.predator = predator
            self.tf_pixel_tensor = tf_pixel_tensor
            self.prey_centers_xy3 = prey_centers_xy3
            self.prediction = self.predator.model.predict(tf_pixel_tensor)[0]
#            self.aim_error = aim_error(self.prediction, self.prey_centers_xy3)
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
            print('starving!!  ',
                  xxx_temp_starvation_count, ', ',
                  worst_predator.step / xxx_temp_starvation_count, ', ',
                  xxx_temp_starvation_count / worst_predator.step)
            # Replace worst predator in Tournament with offspring of other two.
            worst_predator.replace_in_population(self.members[0].predator,
                                                 self.members[1].predator)
            print()


# TODO 20220913 temp ad hoc counter
xxx_temp_starvation_count = 0


# # Run test

# In[8]:


# # TODO 20220827 testing print_model_trace
# test_predator.print_model_trace()



# Keep track of how often selected prey is nearest center:
Predator.nearest_center = 0

# Predator.population = []
# TODO maybe a Predator.reset() method?

# Flush out obsolete files in comms directory.
clean_up_communication_directory()

# Start fresh run defaulting to step 0.
start_run()


# In[ ]:


# Normally start from step 0, or if an "other" file exists
# (eg 'camo_123.jpeg') then restart from that point.

# restart_run()

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789

