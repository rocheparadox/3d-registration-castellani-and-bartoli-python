# Author : Roche Christopher
# 13/07/23 - 11:49:48

import os

PLOTS_DIR = 'plots'


def clean_plots_dir():
    if os.path.exists(PLOTS_DIR):
        for _file in os.listdir(PLOTS_DIR):
            os.remove(os.path.join(PLOTS_DIR, _file))
    else:
        os.mkdir(PLOTS_DIR)
