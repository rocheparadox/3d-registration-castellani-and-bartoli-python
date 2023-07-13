# Author : Roche Christopher
# 10/07/23 - 13:22:44

import os
from moviepy.editor import concatenate_videoclips, ImageClip
PLOTS_DIR = 'plots'


def animate_plots():
    img_files = os.listdir(PLOTS_DIR)
    img_files.sort(key=lambda x: os.path.getmtime(os.path.join(PLOTS_DIR, x)))
    plot_clips = [ImageClip(os.path.join(PLOTS_DIR, plot_img)).set_duration(0.50)
                  for plot_img in img_files]
    concat_clips = concatenate_videoclips(plot_clips, method='compose')
    concat_clips.write_videofile(os.path.join(PLOTS_DIR, 'animated_plot.mp4'), fps=24)
