# -*- coding: utf-8 -*-

import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import Adam

# Relative disk size (diameter) and radius.
fcd_relative_disk_size = 201.0 / 1024.0

# Calculates RELATIVE disk radius on the fly -- rewrite later.
def relative_disk_radius():
    return fcd_relative_disk_size / 2

################################################################################
# Draw utilities
################################################################################

################################################################################
# Draw utilities
################################################################################

# Draw a training image on the log. First arg is either a 24 bit RGB pixel
# representation as read from file, or the rescaled 3xfloat used internally.
# Optionally draw crosshairs to show center of disk.
def draw_image(rgb_pixel_tensor, label=None, prediction=None):
    i24bit = []
    if ((rgb_pixel_tensor.dtype == np.float32) or
        (rgb_pixel_tensor.dtype == np.float64)):
        unscaled_pixels = np.interp(rgb_pixel_tensor, [0, 1], [0, 255])
        i24bit = Image.fromarray(unscaled_pixels.astype('uint8'), mode='RGB')
    else:
        i24bit = Image.fromarray(rgb_pixel_tensor)
    plt.imshow(i24bit)
    image_width = rgb_pixel_tensor.shape[0]
    disk_width = image_width * fcd_relative_disk_size
    if label is not None:
        # width = rgb_pixel_tensor.shape[0]
        # draw_crosshairs(center, width, width * fcd_relative_disk_size)
        # draw_crosshairs(label, width, width * fcd_relative_disk_size)
        draw_crosshairs(label, image_width, disk_width)
    if prediction is not None:
        prediction = np.array(prediction) * image_width
        draw_circle(prediction, disk_width)
    plt.show()

# Draw crosshairs to indicate disk position (label or estimate).
def draw_crosshairs(center, image_size, disk_size):
    m = image_size - 1       # max image coordinate
    s = disk_size * 1.2 / 2  # gap size (radius)
    h = center[0] * m        # center x in pixels
    v = center[1] * m        # center y in pixels
    plt.hlines(v, 0, max(0, h - s), color="black")
    plt.hlines(v, min(m, h + s), m, color="black")
    plt.vlines(h, 0, max(0, v - s), color="white")
    plt.vlines(h, min(m, v + s), m, color="white")

# Draw line in plot between arbitrary points in plot. (Not currently used.)
# eg: draw_line((100, 100), (924, 924), color="yellow")
#     (See https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
#      for another approach)
def draw_line(p1, p2, color="white"):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color)

# Draw a circle, dashed black and white, with given xy "center" and "diameter"
# both specified in pixel units.
def draw_circle(center, diameter):
    center = np.array(center)
    r = diameter / 2
    color = 'black'
    prev_angle = 0
    steps = 24
    for i in range(steps):
        next_angle = (i + 1) * 2 * math.pi / steps
        prev_point = np.array((math.sin(prev_angle), math.cos(prev_angle)))
        next_point = np.array((math.sin(next_angle), math.cos(next_angle)))
        draw_line(center + prev_point * r, center + next_point * r, color)
        if color == 'black':
            color = 'white'
        else:
            color = 'black'
        prev_angle = next_angle

################################################################################
# Keras model utilities
################################################################################

# Construct a Keras model with CNN layers at the front, striding down in
# resolution, then dense layers funneling down to just two output neurons
# representing the predicted image position center of the conspicuous disk.
# (First version cribbed from DLAVA chapter B3, Listing B3-41)

#def make_fcd_cnn_model():
#def make_disk_finder_model():
def make_disk_finder_model(X_train):
    cnn_act = 'relu'
    dense_act = 'relu'
    output_act = 'linear'
    cnn_filters = 32
    cnn_dropout = 0.2
    dense_dropout = 0.5  # ala Hinton (2012)
    input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], )
    #
    model = Sequential()
    # Two "units" of:
    #     CNN: 5x5, 32 filters, relu,
    #     Dropout: 0.2
    #     CNN: 3x3, 32 filters, relu,
    #     Dropout: 0.2
    #     CNN: 3x3, 32 filters, relu, stride down by 2.
    #     Dropout: 0.2
    model.add(Conv2D(cnn_filters, (5, 5), activation=cnn_act, padding='same',
                     kernel_constraint=MaxNorm(3),
                     input_shape=input_shape))
    model.add(Dropout(cnn_dropout))
    model.add(Conv2D(cnn_filters, (3, 3), activation=cnn_act, padding='same',
                     kernel_constraint=MaxNorm(3)))
    model.add(Dropout(cnn_dropout))
    model.add(Conv2D(cnn_filters, (3, 3), activation=cnn_act, padding='same',
                     strides=(2, 2), kernel_constraint=MaxNorm(3)))
    model.add(Dropout(cnn_dropout))
    # Unit 2:
    model.add(Conv2D(cnn_filters, (5, 5), activation=cnn_act, padding='same',
                     kernel_constraint=MaxNorm(3)))
    model.add(Dropout(cnn_dropout))
    model.add(Conv2D(cnn_filters, (3, 3), activation=cnn_act, padding='same',
                     kernel_constraint=MaxNorm(3)))
    model.add(Dropout(cnn_dropout))
    model.add(Conv2D(cnn_filters, (3, 3), activation=cnn_act, padding='same',
                     strides=(2, 2), kernel_constraint=MaxNorm(3)))
    model.add(Dropout(cnn_dropout))
    
    # Then flatten and use a large-ish dense layer with heavy dropout.
    model.add(Flatten())
    model.add(Dense(512, activation=dense_act))
    model.add(Dropout(dense_dropout))

    # Then funnel down to two output neurons for (x, y) of predicted center.
    model.add(Dense(128, activation=dense_act))
    model.add(Dense(32, activation=dense_act))
    model.add(Dense(8, activation=dense_act))
    model.add(Dense(2, activation=output_act))

    # Compile with mse loss, tracking accuracy and fraction-inside-disk.
    model.compile(loss='mse', optimizer='adam', metrics=["accuracy", in_disk])
    return model

# Utility to fit and plot a run, again cribbed from DLAVA chapter B3.
def run_model(model, X_train, y_train, X_test, y_test,
              epochs, batch_size, plot_title):
    history = model.fit(X_train,
                        y_train,
                        validation_data = (X_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size)
    print()
    plot_accuracy_and_loss(history, plot_title)
    return history

# A little utility to draw plots of accuracy and loss.
def plot_accuracy_and_loss(history, plot_title):

    # TODO experiment Skip First Epoch -- do not plot first epoch
    # My FCD losses are so high on the first epoch that it swamps the plot.
    def sfe(e_list):
        return e_list[1:]
    
#    xs = range(len(history.history['accuracy']))
    xs = range(len(sfe(history.history['accuracy'])))
    # plt.figure(figsize=(10,3))
    plt.figure(figsize=(15,3))

    # plt.subplot(1, 2, 1)
    plt.subplot(1, 3, 1)
#    plt.plot(xs, history.history['accuracy'], label='train')
#    plt.plot(xs, history.history['val_accuracy'], label='validation')
    plt.plot(xs, sfe(history.history['accuracy']), label='train')
    plt.plot(xs, sfe(history.history['val_accuracy']), label='validation')
    plt.legend(loc='lower left')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title(plot_title+': Accuracy')

    plt.subplot(1, 3, 2)
#    plt.plot(xs, history.history['in_disk'], label='train')
#    plt.plot(xs, history.history['val_in_disk'], label='validation')
    plt.plot(xs, sfe(history.history['in_disk']), label='train')
    plt.plot(xs, sfe(history.history['val_in_disk']), label='validation')
    plt.legend(loc='lower left')
    plt.xlabel('epochs')
    plt.ylabel('fraction inside disk')
    plt.title(plot_title+': fraction inside disk')

    # plt.subplot(1, 2, 2)
    plt.subplot(1, 3, 3)
#    plt.plot(xs, history.history['loss'], label='train')
#    plt.plot(xs, history.history['val_loss'], label='validation')
    plt.plot(xs, sfe(history.history['loss']), label='train')
    plt.plot(xs, sfe(history.history['val_loss']), label='validation')
    plt.legend(loc='upper left')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(plot_title+': Loss')

    plt.show()

################################################################################
# in_disk metric
################################################################################

# Prototype metric to measure the fraction of predictions that are inside disks.
# For each pair of 2d points of input, output tensor is 1 for IN and 0 for OUT.
# def fcd_prediction_inside_disk(y_true, y_pred):

# (make name shorter so it is easier to read fit() log.)
def in_disk(y_true, y_pred):
    distances = corresponding_distances(y_true, y_pred)
    # relative_disk_radius = (float(fcd_disk_size) / float(fcd_image_size)) / 2

    # From https://stackoverflow.com/a/42450565/1991373
    # Boolean tensor marking where distances are less than relative_disk_radius.
    # insides = tf.less(distances, relative_disk_radius)
#    insides = tf.less(distances, fcd_disk_radius())
    insides = tf.less(distances, relative_disk_radius())
    map_to_zero_or_one = tf.cast(insides, tf.int32)
    return map_to_zero_or_one


#example_true_positions = tf.convert_to_tensor([[1.0, 2.0],
#                                               [3.0, 4.0],
#                                               [5.0, 6.0],
#                                               [7.0, 8.0]])
#example_pred_positions = tf.convert_to_tensor([[1.1, 2.0],
#                                               [3.0, 4.2],
#                                            #    [5.0, 6.1],
#                                               [5.0, 6.0],
#                                               [7.3, 8.0]])
#
# in_disk(example_true_positions, example_pred_positions)
# fcd_disk_shaped_loss_helper(example_true_positions, example_pred_positions)

# Given two tensors of 2d point coordinates, return a tensor of the Cartesian
# distance between corresponding points in the input tensors.
def corresponding_distances(y_true, y_pred):
    true_pos_x, true_pos_y = tf.split(y_true, num_or_size_splits=2, axis=1)
    pred_pos_x, pred_pos_y = tf.split(y_pred, num_or_size_splits=2, axis=1)
    dx = true_pos_x - pred_pos_x
    dy = true_pos_y - pred_pos_y
    distances = tf.sqrt(tf.square(dx) + tf.square(dy))
    return distances

################################################################################
# Miscellaneous utilities
################################################################################

# debug_print('fcd_filename_to_xy_ints("foobar_123_456")')
# debug_print('fcd_normalized_xy("foobar_123_456", np.zeros((1024,1024,3)))')
# debug_print('[123/(1024/input_scale), 456/(1024/input_scale)]')

def timestamp_string():
    # Just assert that we want to use Pacific time, for the benefit of cwr.
    # The Colab server seems to think local time is UTC.
    return datetime.now(pytz.timezone('US/Pacific')).strftime('%Y%m%d_%H%M')

# Reset random sequence seeds in Python's "random", Numpy, and TensorFlow.
random_seed = 12345678
def set_global_random_seed(seed):
    global random_seed
    random_seed = seed
def reset_random_seeds(seed = random_seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

#    # Distance between 2 points in 2d Euclidean space.
#    # (TODO Should generalize to N dimensions)
#    def dist2d(point1, point2):
#        offset = point1 - point2
#        # TODO there has GOT to be a cleaner "more pythonic" way to do this:
#        return math.sqrt(math.pow(offset[0], 2) + math.pow(offset[1], 2))
    
# Distance between 2 points in 2d Euclidean space.
# (TODO Should generalize to N dimensions)
# (After Python 3.8 use math.dist: math.dist([1, 0, 0], [0, 1, 0]))
def dist2d(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# Generic interpolation
# (20220108 borrowed from TexSyn's c++ Utilities package)
def interpolate(alpha, x0, x1):
    return (x0 * (1 - alpha)) + (x1 * alpha)

# # Informal tests:
# print('df.interpolate(0, np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) =',
#       df.interpolate(0, np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])))
# print('df.interpolate(0.5, np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) =',
#       df.interpolate(0.5, np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])))
# print('df.interpolate(1, np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) =',
#       df.interpolate(1, np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])))

# Constrain a given value "x" to be between two bounds: "bound0" and "bound1"
# (without regard to order). Returns x if it is between the bounds, otherwise
# returns the nearer bound.
# (20220108 borrowed from TexSyn's c++ Utilities package)
def clip(x, bound0, bound1):
    clipped = x
    min_bound = min(bound0, bound1)
    max_bound = max(bound0, bound1)
    if (clipped < min_bound):
        clipped = min_bound
    if (clipped > max_bound):
        clipped = max_bound
    return clipped

# # Informal tests:
# print('df.clip(-1, 0, 1) =', df.clip(-1, 0, 1))
# print('df.clip(0, 0, 1) =', df.clip(0, 0, 1))
# print('df.clip(0.5, 0, 1) =', df.clip(0.5, 0, 1))
# print('df.clip(1, 0, 1) =', df.clip(1, 0, 1))
# print('df.clip(2, 0, 1) =', df.clip(2, 0, 1))

# Clip between 0 and 1.
# (20220108 borrowed from TexSyn's c++ Utilities package)
def clip01 (x):
    return clip(x, 0, 1)

# Remap a value specified relative to a pair of bounding values
# to the corresponding value relative to another pair of bounds.
# Inspired by (dyna:remap-interval y y0 y1 z0 z1) circa 1984.
# (20220108 borrowed from TexSyn's c++ Utilities package)
# TODO -- note similar API in numpy
def remapInterval(x, in0, in1, out0, out1):
    # Remap if input range is nonzero, otherwise blend them evenly.
    blend = 0.5
    input_range = in1 - in0;
    if (input_range > 0):
        blend = (x - in0) / input_range
    return interpolate(blend, out0, out1)

# Like remapInterval but the result is clipped to remain between out0 and out1
# (20220108 borrowed from TexSyn's c++ Utilities package)
def remapIntervalClip(x, in0, in1, out0, out1):
    return clip(remapInterval(x, in0, in1, out0, out1), out0, out1)


# Maps from 0 to 1 into a sinusoid ramp ("slow in, slow out") from 0 to 1.
# (20220108 borrowed from TexSyn's c++ Utilities package)
def sinusoid(x):
    return (1 - math.cos(x * math.pi)) / 2;

# print('df.sinusoid (0.00)', df.sinusoid (0.00))
# print('df.sinusoid (0.25)', df.sinusoid (0.25))
# print('df.sinusoid (0.50)', df.sinusoid (0.50))
# print('df.sinusoid (0.75)', df.sinusoid (0.75))
# print('df.sinusoid (1.00)', df.sinusoid (1.00))

# Returns the scalar amplitude of a co-sinusoidal spot, for a given sample
# position, and given spot parameters (center, inner_radius, outer_radius).
# (20220108 borrowed from TexSyn's c++ Utilities package)
def spot_utility(position, center, inner_radius, outer_radius):
    # Distance from sample position to spot center.
    d = dist2d(position, center)
    # Fraction for interpolation: 0 inside, 1 outside, ramp between.
    f = remapIntervalClip(d, inner_radius, outer_radius, 0, 1)
    # map interval [0, 1] to cosine curve.
    return 1 - sinusoid(f)

# # Informal tests:
# print('df.spot_utility(np.array([1,1]), np.array([1,1]), 0, 1) =',
#       df.spot_utility(np.array([1,1]), np.array([1,1]), 0, 1))
# print('df.spot_utility(np.array([1,1]), np.array([0,1]), 0, 1) =',
#       df.spot_utility(np.array([1,1]), np.array([0,1]), 0, 1))
# print('df.spot_utility(np.array([0.5,0.5]), np.array([1,1]), 0, 1) =',
#       df.spot_utility(np.array([0.5,0.5]), np.array([1,1]), 0, 1))

################################################################################
# Visualize labels, or model predictions, of some examples from a given dataset.
################################################################################

#    # Visualize labels, or model predictions, of a random sample of examples from a
#    # given dataset. Must pass in a tensor of image tensors, one of EITHER a tensor
#    # of labels (as xy of disk centers) OR a trained model for making predictions
#    # from an image. Plus optionally a count of how many random examples to draw.
#    # The labels (or predictions) are shown as overlaid crosshairs.
#    def visualize_dataset(images, labels=None, model=None, count=10):
#        for i in range(count):
#            r = random.randrange(0, images.shape[0])
#            pixel_tensor = images[r, :, :, :]
#            center = (0, 0)
#            if model is None:
#                center = labels[r, :]
#            if labels is None:
#                center = model.predict(tf.convert_to_tensor([pixel_tensor]))[0]
#            print(r, ": (", center[0], ",", center[1], ")")
#            draw_image(pixel_tensor, center)

# Visualize labels, or model predictions, of a random sample of examples from a
# given dataset. Must pass in a tensor of image tensors, one of EITHER a tensor
# of labels (as xy of disk centers) OR a trained model for making predictions
# from an image. Plus optionally a count of how many random examples to draw.
# The labels (or predictions) are shown as overlaid crosshairs.
def visualize_dataset(images, labels=None, model=None, count=10):
    for i in range(count):
        r = random.randrange(0, images.shape[0])
        pixel_tensor = images[r, :, :, :]
#        center = (0, 0)
        label = None
        prediction = None
#        if model is None:
#            center = labels[r, :]
#        if labels is None:
#            center = model.predict(tf.convert_to_tensor([pixel_tensor]))[0]
        if labels is not None:
            label = labels[r, :]
        if model is not None:
            prediction = model.predict(tf.convert_to_tensor([pixel_tensor]))[0]
#        print(r, ": (", center[0], ",", center[1], ")")
        print(r)
#        draw_image(pixel_tensor, center)
        draw_image(pixel_tensor, label, prediction)

################################################################################
# Soft matte image compositing tool.
################################################################################

## Matte (composite) one "foreground" RGB image over another "background" image,
## according to per-pixel scalar alpha weights in "matte".
#def soft_matte(foreground, background, matte):
#
#    # Multiply given (RGB) image by weights in given (single channel) matte.
#    def weighted_element(image, matte):
#        w = image.shape[0]
#        h = image.shape[1]
#        reshape_image = image.reshape(w * h, 3).T
#        reshape_matte = matte.reshape(w * h, 1).T
#        alpha_reshape = reshape_matte * reshape_image
#        alpha = alpha_reshape.T.reshape([w, h, 3])
#        return alpha
#
#    # Intended as more specific error messages for mismatch of args.
#    def check_args(foreground, background, matte):
#        assert foreground.shape[2] == 3, "'foreground' should be 3 floats deep"
#        assert background.shape[2] == 3, "'background' should be 3 floats deep"
#        assert matte.shape[2] == 1, "'matte' should be 1 float deep"
#        a = foreground.shape[0:1] == matte.shape[0:1]
#        assert a, "width or height of 'foreground' and 'matte' do not match"
#        a = background.shape[0:1] == matte.shape[0:1]
#        assert a, "width or height of 'background' and 'matte' do not match"
#
#    # Check args, then sum two image arguments, weighted by matte and inverse.
#    check_args(foreground, background, matte)
#    return (weighted_element(foreground, matte) +
#            weighted_element(background, 1 - matte))

# Matte (composite) one "foreground" RGB image over another "background" image,
# according to per-pixel scalar alpha weights in "matte".
def soft_matte(foreground, background, matte):
    check_args_for_soft_matte(foreground, background, matte)
    # Sum the two images, weighted by matte and its inverse.
    return (weighted_element(foreground, matte) +
            weighted_element(background, 1 - matte))
            
# Multiply each pixel of an (RGB) image by weights in a (single channel) matte.
def weighted_element(image, matte):
    w = image.shape[0]
    h = image.shape[1]
    reshape_image = image.reshape(w * h, 3).T
    reshape_matte = matte.reshape(w * h, 1).T
    alpha_reshape = reshape_matte * reshape_image
    alpha = alpha_reshape.T.reshape([w, h, 3])
    return alpha

# Intended as more specific error messages for mismatch of args.
def check_args_for_soft_matte(foreground, background, matte):
    assert foreground.shape[2] == 3, "'foreground' should be 3 floats deep"
    assert background.shape[2] == 3, "'background' should be 3 floats deep"
    assert matte.shape[2] == 1, "'matte' should be 1 float deep"
    a = foreground.shape[0:1] == matte.shape[0:1]
    assert a, "width or height of 'foreground' and 'matte' do not match"
    a = background.shape[0:1] == matte.shape[0:1]
    assert a, "width or height of 'background' and 'matte' do not match"

################################################################################
# Generate a matte for "soft edged spot".
################################################################################

# Generate a soft disk matte, an array with shape (diameter, diameter, 1).
# Uses the TexSyn Spot parameters: inner_radius and outer_radius.
def make_disk_matte(diameter, inner_radius, outer_radius):
    # Coerce diameter to be int, to be used as an index for pixels.
    diameter = round(diameter)
    # Make the array to return.
    image = np.zeros((diameter, diameter, 1), dtype=np.float32)
    # Centerpoint of disk (as float, independent of pixel boundaries).
    cp = (diameter / 2.0, diameter / 2.0)
    # Loop over all pixel coordinates (xi, yi) with (xf, yf) as center of pixel.
    for xi in range(diameter):
        xf = xi + 0.5
        for yi in range(diameter):
            yf = yi + 0.5
            image[xi, yi, :] = spot_utility((xf, yf), cp,
                                            inner_radius, outer_radius)
    return image

################################################################################
# Low pass filtering.
################################################################################

def lpf_3x3_box(image):
    s = image.shape
    filtered = np.empty(s, dtype=np.float32)
    for x in range(s[0]):
        for y in range(s[1]):
            sum = [0,0,0]
            count = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    p = x + i
                    q = y + j
                    if (p >= 0) and (q >= 0) and (p < s[0]) and (q < s[1]):
                        count += 1
                        sum += image[p, q, :]
            filtered[x, y, :] = sum / count
    return filtered

def lpf_3x3(image):
    s = image.shape
    filtered = np.empty(s, dtype=np.float32)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])
    for x in range(s[0]):
        for y in range(s[1]):
            sum = [0,0,0]
            count = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    p = x + i
                    q = y + j
                    if (p >= 0) and (q >= 0) and (p < s[0]) and (q < s[1]):
                        # count += 1
                        # sum += image[p, q, :]
                        w = kernel[i - 1, j - 1]
                        count += w
                        sum += image[p, q, :] * w
            filtered[x, y, :] = sum / count
    return filtered

################################################################################
# Checkerboard test pattern.
################################################################################

# Generate image (rgb pixel tensor) with a black and white checkerboard pattern.
def make_checkerboard(size, half_wavelength, c0=[0,0,0], c1=[1,1,1]):
    # image = np.zeros((size, size, 3), dtype=np.float32)
    image = np.empty((size, size, 3), dtype=np.float32)
    image[:,:,:] = c0
    for x in range(size):
        for y in range(size):
            if 0 == ((int(x / half_wavelength) + int(y / half_wavelength)) % 2):
                # image[x, y, :] = [1, 1, 1]
                image[x, y, :] = c1
    return image

################################################################################
# Read label (disk centerpoint) encoded in FCD filename.
################################################################################

# Parse FCD filename to a list of two ints: (x, y) pixel coordinates.
def fcd_filename_to_xy_ints(filename):
    without_extension = filename.split('.')[0]
    two_numeric_strings = without_extension.split('_')[1:3]
    return list(map(int, two_numeric_strings))

# Get image label from image file names ([x, y] as floats on [0,1])
# def fcd_normalized_xy(filename, pixels):
def fcd_normalized_xy(filename, pixels, input_scale = 1):
    pixel_coordinates = fcd_filename_to_xy_ints(filename)
    return pixel_coordinates / (np.array(pixels.shape)[1:2] / input_scale)

################################################################################
