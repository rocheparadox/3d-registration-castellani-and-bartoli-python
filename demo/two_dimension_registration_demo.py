# Author : Roche Christopher
# 13/07/23 - 11:46:25

from castellani_bartoli.castellani_bartoli import registration
from utils.file_utils import clean_plots_dir
from utils.utils import *
from utils.plot_utils import animate_plots

if __name__ == '__main__':
    # empty plots directory
    print('Cleaning plots dir')
    clean_plots_dir()
    print('Done cleaning plots directory')

    # Create a 3-D point set
    model = np.transpose(np.array([[1, 1, 0], [3, 3, 0], [5, 5, 0], [7, 6, 0], [8, 5, 0], [9, 4, 0], [11, 5, 0],
                                   [13, 7, 0], [15, 12, 0], [16, 14, 0], [18, 13, 0], [20, 13, 0], [21, 7, 0],
                                   [23, 5, 0], [24, 3, 0], [25, 1, 0], [26, 0, 0], [28, -3, 0], [29, -4, 0],
                                   [30, -7, 0], [31, -8, 0], [33, -10, 0], [34, -9, 0], [35, -11, 0], [37, -13, 0],
                                   [38, -14, 0]]))
    print(model.shape)
    altered_data = model
    # Rotate it for -45 degrees(clockwise) with respect to z axis
    altered_data = rotate_matrix(altered_data, 297)
    # Translate it
    altered_data = translate_matrix(altered_data, 23, 69, 0)
    altered_data = rotate_matrix(altered_data, 12)
    # altered_data = translate_matrix(altered_data, 30, 0, 0)
    # Now we have two 3-D point sets.

    registration(model, altered_data)
    # animate the plot images
    animate_plots()
    exit()
