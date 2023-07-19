# Author : Roche Christopher
# 13/07/23 - 19:15:43

import sys

sys.path.insert(1, '../')

from plyfile import PlyData, PlyElement

from castel_bart_3D_registration.castellani_bartoli.castellani_bartoli import registration
from castel_bart_3D_registration.utils.file_utils import clean_plots_dir
from castel_bart_3D_registration.utils.utils import *
from castel_bart_3D_registration.utils.plot_utils import animate_plots, plot_3d, plot_3d_model_and_data


STANFORD_BUNNY_POINTCLOUD_FILE = 'bunny/reconstruction/bun_zipper_res3.ply'
use_bunny_model = True


def get_stanford_bunny_poincloud():
    print(f'Loading bunny data from {STANFORD_BUNNY_POINTCLOUD_FILE}')
    bunny_plydata = PlyData.read(STANFORD_BUNNY_POINTCLOUD_FILE)
    rows = bunny_plydata.elements[0].data.shape[0]
    print(f'There are {rows} number of points in the point cloud')
    bunny_pointcloud = np.zeros((3, rows))
    bunny_pointcloud[0, :] = bunny_plydata.elements[0].data['x']
    bunny_pointcloud[1, :] = bunny_plydata.elements[0].data['y']
    bunny_pointcloud[2, :] = bunny_plydata.elements[0].data['z']
    print(f'Bunny data loaded')
    return bunny_pointcloud


if __name__ == '__main__':
    # empty plots directory
    print("-- Three Dimensional Registration Demo -- ")
    print('Cleaning plots dir')
    clean_plots_dir()
    print('Done cleaning plots directory')

    if use_bunny_model:
        model = get_stanford_bunny_poincloud()
    else:
        # Create a 3-D point set
        model = np.transpose(np.array([[1, 1, 0], [3, 3, 0], [5, 5, 0], [7, 6, 0], [8, 5, 0], [9, 4, 0], [11, 5, 0],
                                       [13, 7, 0], [15, 12, 0], [16, 14, 0], [18, 13, 0], [20, 13, 0], [21, 7, 0],
                                       [23, 5, 0], [24, 3, 0], [25, 1, 0], [26, 0, 0], [28, -3, 0], [29, -4, 0],
                                       [30, -7, 0], [31, -8, 0], [33, -10, 0], [34, -9, 0], [35, -11, 0], [37, -13, 0],
                                       [38, -14, 0]]))

    altered_data = model

    altered_data = rotate_matrix(altered_data, 23, axis='x')
    altered_data = rotate_matrix(altered_data, 13, axis='y')
    altered_data = translate_matrix(altered_data, 15, 12, 16)

    # plot_3d_model_and_data(model, altered_data)

    rotation_matrix, translational_matrix = registration(model, altered_data, save_figure=True, show_plot=False)
    print(f"The icp rotation and translation matrices are {rotation_matrix} and {translational_matrix}")
    # redefined_view = apply_transformation(altered_data, rotation_matrix, translational_matrix)
    #
    # plot_3d_model_and_data(model, redefined_view)
    # plot_3d_model_and_data(model, redefined_view)
    # animate the plot images
    animate_plots()
    exit()
