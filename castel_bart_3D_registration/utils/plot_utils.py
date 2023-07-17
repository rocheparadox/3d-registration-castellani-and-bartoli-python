# Author : Roche Christopher
# 13/07/23 - 11:45:22

import os

import numpy
from matplotlib import pyplot
from moviepy.editor import concatenate_videoclips, ImageClip

from .file_utils import PLOTS_DIR


def plot_2d_model_and_data(model_view, data_view, save_figure=False, figure_name='plot.png', title=None, dpi=200):
    pyplot.axis([-20, 50, -40, 40])
    pyplot.scatter(model_view[0,], model_view[1,])
    pyplot.plot(model_view[0,], model_view[1,])

    pyplot.scatter(data_view[0,], data_view[1,])
    pyplot.plot(data_view[0,], data_view[1,])
    if title is not None:
        pyplot.title(title)
    if save_figure:
        pyplot.savefig(figure_name, dpi=dpi)
    else:
        pyplot.show()
    pyplot.clf()  # clear the plot


def plot_3d_model_and_data(model_view, data_view, save_figure=False, figure_name='plot.png', title=None, dpi=200):
    fig = pyplot.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.axes.set_xlim3d(-30, 30)
    ax.axes.set_ylim3d(0, 60)
    ax.axes.set_zlim3d(-30, 30)

    for datapoint in numpy.transpose(model_view):
        ax.scatter(datapoint[0], datapoint[1], datapoint[2], color='blue')
        ax.plot(datapoint[0], datapoint[1], datapoint[2])
    for datapoint in numpy.transpose(data_view):
        ax.scatter(datapoint[0], datapoint[1], datapoint[2], color='orange')
        ax.plot(datapoint[0], datapoint[1], datapoint[2])
    if title is not None:
        pyplot.title(title)
    if not save_figure:
        pyplot.show()
    else:
        elev = 10
        azim = -40
        roll = 0
        ax.view_init(elev, azim, roll, vertical_axis='z')
        pyplot.savefig(figure_name, dpi=dpi)
    pyplot.clf()


def plot_3d(data, color='blue'):
    fig = pyplot.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    for datapoint in numpy.transpose(data):
        ax.scatter(datapoint[0], datapoint[1], datapoint[2], color=color)
    pyplot.show()


def animate_plots():
    img_files = os.listdir(PLOTS_DIR)
    img_files.sort(key=lambda x: os.path.getmtime(os.path.join(PLOTS_DIR, x)))
    plot_clips = [ImageClip(os.path.join(PLOTS_DIR, plot_img)).set_duration(0.50)
                  for plot_img in img_files]
    concat_clips = concatenate_videoclips(plot_clips, method='compose')
    concat_clips.write_videofile(os.path.join(PLOTS_DIR, 'animated_plot.mp4'), fps=24)