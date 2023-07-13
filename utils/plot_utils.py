# Author : Roche Christopher
# 13/07/23 - 11:45:22

import os
from matplotlib import pyplot
from moviepy.editor import concatenate_videoclips, ImageClip

from utils.file_utils import PLOTS_DIR


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


def animate_plots():
    img_files = os.listdir(PLOTS_DIR)
    img_files.sort(key=lambda x: os.path.getmtime(os.path.join(PLOTS_DIR, x)))
    plot_clips = [ImageClip(os.path.join(PLOTS_DIR, plot_img)).set_duration(0.50)
                  for plot_img in img_files]
    concat_clips = concatenate_videoclips(plot_clips, method='compose')
    concat_clips.write_videofile(os.path.join(PLOTS_DIR, 'animated_plot.mp4'), fps=24)