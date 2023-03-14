#!/usr/bin/env python
# coding: utf-8
################################################################################
#
# PredatorServer.py -- PredatorEye system
#
# Manages the "predator side" of the advarsarial camouflage evolution simulation
#
# 20220924 Split off from EvoCamoVsLearnPredPop.py (from ...ipynb)
# Copyright © 2022 Craig Reynolds. All rights reserved.
#
################################################################################


import time
import numpy as np
import DiskFind as df
import tensorflow as tf

from pathlib import Path
from Predator import Predator
from Tournament import Tournament

shared_directory = None
my_prefix = "find_"
other_prefix = "camo_"
my_suffix =  ".txt"
other_suffix = ".png"


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
    print('\nstep', step)
    wait_for_reply(step, shared_directory)
    write_response_file(step, shared_directory)
    delete_find_file(step - 1, shared_directory)

# Read image file, loop if not expected format. (That is: assume sync delay.)
def patiently_read_pixel_tensor(step, directory):
    pixel_tensor = None
    image_pathname = make_camo_pathname(step, directory)
    while True:
        image_read_ok = True
        try:
            pixel_tensor = df.read_image_file_as_pixel_tensor(image_pathname)
        except:
            image_read_ok = False
        if image_read_ok and df.check_pixel_tensor(pixel_tensor):
            break
        print('Reread: bad format for', image_pathname)
        time.sleep(0.1)
    return pixel_tensor

# Read image file for step, apply pre-trained model, write response file.
def write_response_file(step, directory):
    # Read image file and check for expected format.
    pixel_tensor = patiently_read_pixel_tensor(step, directory)
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
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO 20230130 make FineTuningDataset into a class
#    ftd.update(pixel_tensor, best_prediction, prey_centers_xy3, step, directory)
#    Predator.ftd.update(pixel_tensor, best_prediction,
#                        prey_centers_xy3, step, directory)
#    # Update the fine-tuning dataset of each Predator in the current tournament.
#    for p in tournament.members:
#        p.ftd.update(pixel_tensor, best_prediction, prey_centers_xy3)
    # Update the fine-tuning dataset of each Predator in the current tournament.
    for m in tournament.members:
        m.predator.ftd.update(pixel_tensor, best_prediction, prey_centers_xy3)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Update the Predator population in case of starvation.
    Predator.step = step  # TODO 20220911 Goofy hack.
    tournament.update_predator_population()

    # Fine-tune models of each Predator in Tournament.
    tournament.fine_tune_models()
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TODO 20230309 process any SQM files

# Before waiting for the next "camo_" file, this checks for and processes all
# "eval_" files for Static Quality Metric. If it finds any, it holds control
# until they are all handled.

def process_any_sqm_files(step, directory):
    directory_path = Path(directory)
    sqm_index = -1
    
    # Form pathname for "preprefix_index_step.txt" response file.
    def make_sqm_pathname(preprefix, index):
        return directory + preprefix + str(index) + '_' + str(step) + ".txt"

    def make_eval_pathname(index):
        return directory + "eval_" + str(index) + '_' + str(step) + ".png"

    # Delete the given file, presumably after having written the next one.
    def delete_previous_sqm_file(index, step, directory):
        p = Path(make_sqm_pathname('sqm_', index - 1))
        if p.exists():
            p.unlink()

#    # Write response file (contains an xy position as 2 floats on [0,1]).
#    def write_sqm_response():
#        response_string = '0.5 0.5'
#        verify_comms_directory_reachable()
#        wait_pathname = Path(make_sqm_pathname('wait_', sqm_index))
#        done_pathname = Path(make_sqm_pathname('sqm_', sqm_index))
#        with open(wait_pathname, 'w') as file:
#            file.write(response_string)
#        wait_pathname.rename(done_pathname)
#        print('Wrote ' + "'" + response_string + "'",
#              'to response file', done_pathname.name)
#        delete_previous_sqm_file(sqm_index, step, directory)

    # Write response file (contains an xy position as 2 floats on [0,1]).
    def write_sqm_response(xy_prediction):
#        response_string = '0.5 0.5'
        response_string = str(xy_prediction[0]) + ' ' + str(xy_prediction[1])
        verify_comms_directory_reachable()
        wait_pathname = Path(make_sqm_pathname('wait_', sqm_index))
        done_pathname = Path(make_sqm_pathname('sqm_', sqm_index))
        with open(wait_pathname, 'w') as file:
            file.write(response_string)
        wait_pathname.rename(done_pathname)
        print('Wrote ' + "'" + response_string + "'",
              'to response file', done_pathname.name)
        delete_previous_sqm_file(sqm_index, step, directory)

    def get_start_index():
        for path in directory_path.iterdir():
            name = str(path.name)
            preprefix = 'eval_'
            ppfl = len(preprefix)
            if preprefix == name[0:ppfl]:
                file_number_string = name[ppfl:name[ppfl:].find('_') + ppfl]
                return int(file_number_string)
    
    def xy_prediction_from_image(image_pathname):
        model = Predator.default_pre_trained_model
        pixel_tensor = df.read_image_file_as_pixel_tensor(image_pathname)
        tf_pixel_tensor = tf.convert_to_tensor([pixel_tensor])
        prediction = model.predict(tf_pixel_tensor, verbose=0)[0]
#        print("prediction =", prediction)
#        return [0, 0]
        return prediction


    # Main loop to process any SQM requests until there are none.
    done = False
    while not done:
        done = True

        if sqm_index < 0:
            i = get_start_index()
            if type(i) == int:
                sqm_index = i
                print('sqm_index =', sqm_index)

        if sqm_index >= 0:
            print('waiting for', Path(make_eval_pathname(sqm_index)).name)
            
        image_pathname = make_eval_pathname(sqm_index)
#        if is_file_present(make_eval_pathname(sqm_index)):
        if is_file_present(image_pathname):
            done = False
#            print('Found', Path(make_eval_pathname(sqm_index)).name)
            print('Found', Path(image_pathname).name)
            
            xy_prediction = xy_prediction_from_image(image_pathname)
            print("xy_prediction =", xy_prediction)

            write_sqm_response(xy_prediction)
            sqm_index += 1
            
#        delete_previous_sqm_file(sqm_index, step, directory)

#        print("Predator.default_pre_trained_model =",
#              Predator.default_pre_trained_model)


        

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


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
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # TODO 20230309 process any SQM files
        process_any_sqm_files(step, directory)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # TODO 20230204 1 sec was OK when communicating with a cloud server and
        #     round trip times were 40 seconds. Now in the per-predator-FTD era,
        #     step times are 1-2 seconds and this needs to be faster.
#        time.sleep(1)
        time.sleep(0.1)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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
