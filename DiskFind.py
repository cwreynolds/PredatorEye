# -*- coding: utf-8 -*-

import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pytz
from datetime import datetime

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

# Draw a training image on the log. First arg is either a 24 bit RGB pixel
# representation as read from file, or the rescaled 3xfloat used internally.
# Optionally draw crosshairs to show center of disk.
def draw_image(rgb_pixel_tensor, center=(0,0)):
    i24bit = []
    if ((rgb_pixel_tensor.dtype == np.float32) or
        (rgb_pixel_tensor.dtype == np.float32)):
        unscaled_pixels = np.interp(rgb_pixel_tensor, [0, 1], [0, 255])
        i24bit = Image.fromarray(unscaled_pixels.astype('uint8'), mode='RGB')
    else:
        i24bit = Image.fromarray(rgb_pixel_tensor)
    plt.imshow(i24bit)
    if ((center[0] != 0) or (center[1] != 0)):
        width = rgb_pixel_tensor.shape[0]
        draw_crosshairs(center, width, width * fcd_relative_disk_size)
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
              fcd_epochs, fcd_batch_size, plot_title):
    history = model.fit(X_train,
                        y_train,
                        validation_data = (X_test, y_test),
                        epochs=fcd_epochs,
                        batch_size=fcd_batch_size)
    print()
    plot_accuracy_and_loss(history, plot_title)
    return history

# A little utility to draw plots of accuracy and loss.
def plot_accuracy_and_loss(history, plot_title):
    xs = range(len(history.history['accuracy']))
    # plt.figure(figsize=(10,3))
    plt.figure(figsize=(15,3))

    # plt.subplot(1, 2, 1)
    plt.subplot(1, 3, 1)
    plt.plot(xs, history.history['accuracy'], label='train')
    plt.plot(xs, history.history['val_accuracy'], label='validation')
    plt.legend(loc='lower left')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title(plot_title+': Accuracy')

    plt.subplot(1, 3, 2)
    # plt.plot(xs, history.history['fcd_prediction_inside_disk'], label='train')
    # plt.plot(xs, history.history['val_fcd_prediction_inside_disk'], label='validation')
    plt.plot(xs, history.history['in_disk'], label='train')
    plt.plot(xs, history.history['val_in_disk'], label='validation')
    plt.legend(loc='lower left')
    plt.xlabel('epochs')
    plt.ylabel('fraction inside disk')
    plt.title(plot_title+': fraction inside disk')

    # plt.subplot(1, 2, 2)
    plt.subplot(1, 3, 3)
    plt.plot(xs, history.history['loss'], label='train')
    plt.plot(xs, history.history['val_loss'], label='validation')
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


# 20220108 WIP reference code from TexSyn

#    // Generic interpolation
#    template<typename F,typename T>
#    T interpolate(const F& alpha, const T& x0, const T& x1)
#    {
#        return (x0 * (1 - alpha)) + (x1 * alpha);
#    }

# Generic interpolation
def interpolate(alpha, x0, x1):
    return (x0 * (1 - alpha)) + (x1 * alpha)


#    // Constrain a given value "x" to be between two bounds: "bound0" and "bound1"
#    // (without regard to order). Returns x if it is between the bounds, otherwise
#    // returns the nearer bound.
#    inline float clip(float x, float bound0, float bound1)
#    {
#        float clipped = x;
#        float min = std::min(bound0, bound1);
#        float max = std::max(bound0, bound1);
#        if (clipped < min) clipped = min;
#        if (clipped > max) clipped = max;
#        return clipped;
#    }

# Constrain a given value "x" to be between two bounds: "bound0" and "bound1"
# (without regard to order). Returns x if it is between the bounds, otherwise
# returns the nearer bound.
def clip(x, bound0, bound1):
    clipped = x
    min_bound = min(bound0, bound1)
    max_bound = max(bound0, bound1)
    if (clipped < min_bound):
        clipped = min_bound
    if (clipped > max_bound):
        clipped = max_bound
    return clipped

#    inline float clip01 (const float x)
#    {
#        return clip(x, 0, 1);
#    }

def clip01 (x):
    return clip(x, 0, 1)

#    // Remap a value specified relative to a pair of bounding values
#    // to the corresponding value relative to another pair of bounds.
#    // Inspired by (dyna:remap-interval y y0 y1 z0 z1) circa 1984.
#    inline float remapInterval(float x,
#                               float in0, float in1,
#                               float out0, float out1)
#    {
#        // Remap if input range is nonzero, otherwise blend them evenly.
#        float input_range = in1 - in0;
#        float blend = ((input_range > 0) ? ((x - in0) / input_range) : 0.5);
#        return interpolate(blend, out0, out1);
#    }

# TODO -- note similar API in numpy
# Remap a value specified relative to a pair of bounding values
# to the corresponding value relative to another pair of bounds.
# Inspired by (dyna:remap-interval y y0 y1 z0 z1) circa 1984.
def remapInterval(x, in0, in1, out0, out1):
    # Remap if input range is nonzero, otherwise blend them evenly.
    blend = 0.5
    input_range = in1 - in0;
    if (input_range > 0):
        blend = (x - in0) / input_range
    return interpolate(blend, out0, out1)


#    // Like remapInterval but the result is clipped to remain between out0 and out1
#    inline float remapIntervalClip(float x,
#                                   float in0, float in1,
#                                   float out0, float out1)
#    {
#        return clip(remapInterval(x, in0, in1, out0, out1), out0, out1);
#    }

# Like remapInterval but the result is clipped to remain between out0 and out1
def remapIntervalClip(x, in0, in1, out0, out1):
    return clip(remapInterval(x, in0, in1, out0, out1), out0, out1)



#    // Maps from 0 to 1 into a sinusoid ramp ("slow in, slow out") from 0 to 1.
#    inline float sinusoid (float x)
#    {
#        return (1 - std::cos(x * M_PI)) / 2;
#    }


# Maps from 0 to 1 into a sinusoid ramp ("slow in, slow out") from 0 to 1.
def sinusoid (x):
    return (1 - math.cos(x * math.pi)) / 2;


#    // Returns the scalar amplitude of a co-sinusoidal spot, for a given sample
#    // position, and given spot parameters (center, inner_radius, outer_radius)
#    // as in Spot::getColor(), etc. (TODO use it there?)
#    float spot_utility(Vec2 position,
#                       Vec2 center,
#                       float inner_radius,
#                       float outer_radius)
#    {
#        // Distance from sample position to spot center.
#        float d = (position - center).length();
#        // Fraction for interpolation: 0 inside, 1 outside, ramp between.
#        float f = remapIntervalClip(d, inner_radius, outer_radius, 0, 1);
#        // map interval [0, 1] to cosine curve.
#        return 1 - sinusoid(f);
#    }


def dist2d(point1, point2):
    offset = point1 - point2
    # TODO there has GOT to be a cleaner "more pythonic" way to do this:
    return math.sqrt(math.pow(offset[0], 2) + math.pow(offset[1], 2))

# Returns the scalar amplitude of a co-sinusoidal spot, for a given sample
# position, and given spot parameters (center, inner_radius, outer_radius).
def spot_utility(position, center, inner_radius, outer_radius):
    # Distance from sample position to spot center.
#    d = (position - center).length();
    d = dist2d(position, center)
    # Fraction for interpolation: 0 inside, 1 outside, ramp between.
    f = remapIntervalClip(d, inner_radius, outer_radius, 0, 1)
    # map interval [0, 1] to cosine curve.
    return 1 - sinusoid(f)

################################################################################
