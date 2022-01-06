# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pytz
from datetime import datetime

# 20220104 this is wrong!
# It duplicates the definition in Find_Conspicuous_Disk.ipynb
# But doing it temporarily to help test this file DiskFind.py
#fcd_image_size = 1024
#fcd_disk_size = 201

# This is only slightly better.
fcd_relative_disk_size = 201.0 / 1024.0

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
def make_disk_finder_model():
    cnn_act = 'relu'
    dense_act = 'relu'
    output_act = 'linear'
    cnn_filters = 32
    cnn_dropout = 0.2
    dense_dropout = 0.5  # ala Hinton (2012)
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
                     input_shape=(fcd_image_size, fcd_image_size, 3)))
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
#def run_model(model_maker, plot_title):
#    model = model_maker()
def run_model(model, plot_title):

    # print("In run_model():")
    # debug_print('X_train.shape')
    # debug_print("y_train.shape")
    # 20211218
    # history = model.fit(X_train, y_train, validation_split=0.2,
    #                     epochs=fcd_epochs, batch_size=fcd_batch_size)
    
    history = model.fit(X_train,
                        y_train,

                        # validation_split=0.2,
                        validation_data = (X_test, y_test),
                        
                        epochs=fcd_epochs,
                        batch_size=fcd_batch_size)


    print()
    plot_accuracy_and_loss(history, plot_title)
    return model, history

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
# Miscellaneous utilities
################################################################################

# debug_print('fcd_filename_to_xy_ints("foobar_123_456")')
# debug_print('fcd_normalized_xy("foobar_123_456", np.zeros((1024,1024,3)))')
# debug_print('[123/(1024/input_scale), 456/(1024/input_scale)]')

def timestamp_string():
    # Just assert that we want to use Pacific time, for the benefit of cwr.
    # The Colab server seems to think local time is UTC.
    return datetime.now(pytz.timezone('US/Pacific')).strftime('%Y%m%d_%H%M')
