# Data Visulizer script for Navier-Stokes equation
# Author: Raj Sutariya
import matplotlib.pyplot as plt
from utilities3 import *
from matplotlib.widgets import Slider, Button

NV = False
DARCY = False
HEAT = True

if NV:
    # read the data
    data2d_t = MatReader('data/ns_V1e-3_N5000_T50.mat')
    test = data2d_t.read_field('u')

    # plot the data
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    plot_data = test[0,:,:,0]
    plt.imshow(test[0,:,:,0], interpolation='nearest', cmap='rainbow',
               origin='lower', aspect='auto')
    plt.colorbar()

    # time and sample slider
    axtime = plt.axes([0.15, 0.15, 0.65, 0.03])
    axsmpl = plt.axes([0.15, 0.1, 0.65, 0.03])

    time_slide = Slider(axtime, 'Time', 0, 50, 0)
    sample_slide = Slider(axsmpl, 'Sample', 0, 5000, 0)
    def update(val):
        time = int(time_slide.val)
        sample = int(sample_slide.val)
        ax.imshow(test[sample, :, :, time], interpolation='nearest', cmap='rainbow',
               origin='lower', aspect='auto')
    time_slide.on_changed(update)
    sample_slide.on_changed(update)

    # pull the trigger
    plt.show()


if DARCY:
    # read the data
    data2d = scipy.io.loadmat('data/piececonst_r421_N1024_smooth1.mat')
    test = data2d['sol']

    # plot the data
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    plot_data = test[0, :, :]
    plt.imshow(test[0, :, :], interpolation='nearest', cmap='rainbow',
               origin='lower', aspect='auto')
    plt.colorbar()

    # time and sample slider
    axsmpl = plt.axes([0.15, 0.1, 0.65, 0.03])

    sample_slide = Slider(axsmpl, 'Sample', 0, 1024, 0)


    def update(val):
        sample = int(sample_slide.val)
        ax.imshow(test[sample, :, :], interpolation='nearest', cmap='rainbow',
                  origin='lower', aspect='auto')

    sample_slide.on_changed(update)

    # pull the trigger
    plt.show()

if HEAT:
    # read the data
    data2d = scipy.io.loadmat('data/heat_N1100_T200_r200.mat')
    test = data2d['u']

    # plot the data
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    plot_data = test[0, :, :]
    plt.imshow(test[0, :, :], interpolation='nearest', cmap='rainbow',
               origin='lower', aspect='auto')
    plt.colorbar()

    # time and sample slider
    axsmpl = plt.axes([0.15, 0.1, 0.65, 0.03])

    sample_slide = Slider(axsmpl, 'Sample', 0, 1100, 0)


    def update(val):
        sample = int(sample_slide.val)
        ax.imshow(test[sample, :, :], interpolation='nearest', cmap='rainbow',
                  origin='lower', aspect='auto')

    sample_slide.on_changed(update)

    # pull the trigger
    plt.show()


