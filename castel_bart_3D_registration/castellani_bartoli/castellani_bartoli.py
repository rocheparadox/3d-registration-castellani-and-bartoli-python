# Author : Roche Christopher
# 19/06/23 - 17:16:12

# The following code contains implementation of the work on 3d registration by Umberto Castellani and Adrien Bartoli
# http://encov.ip.uca.fr/publications/pubfiles/2020_Castellani_etal_3DIAA_registration.pdf
import os.path

import numpy as np
from castel_bart_3D_registration.utils.utils import calculate_centroid, calculate_euclidean_distance, calculate_cross_covariance_matrix, \
    apply_transformation
from castel_bart_3D_registration.utils.file_utils import PLOTS_DIR
from castel_bart_3D_registration.utils import plot_utils

max_iterations = 150


def error_icp(altered_data, correspondences):
    # reshape the altered data and correspondences
    altered_data = np.transpose(altered_data)
    correspondences = np.transpose(correspondences)

    sse = 0 # Squared sum of error
    #print('------inside error_icp-------------')
    #print(f'The altered data shape is {altered_data.shape} and its correspondences shape is {correspondences.shape}')
    for index, data in enumerate(altered_data):
        sse += calculate_euclidean_distance(data, correspondences[index])

    return sse


def registration(model_view, data_view, show_plot=False, save_figure=True, two_dimension=False):

    ultimate_rotational_matrix = np.zeros((3, 3))
    ultimate_translational_matrix = np.zeros((3, 1))

    # Calculate centroids
    model_view_mean = calculate_centroid(model_view)
    data_view_mean = calculate_centroid(data_view)
    print(f'The means of model and data are {model_view_mean} and {data_view_mean} respectively')

    # Move the centroid of data view to the centroid of model view
    # mean_difference = altered_data_mean - model_mean
    # print(f'Altered data is {altered_data}')
    # altered_data = altered_data - mean_difference
    # print(f'Altered data after moving the centroid is {altered_data}')

    correspondences = np.empty(model_view.shape)
    # Find correspondence points using the euclidean distance
    for index, datapoint in enumerate(np.transpose(data_view)):
        closest_point = None
        minimum_distance = None
        #print(f'To find the distance for datapoint {datapoint}')
        for model_point in np.transpose(model_view):
            distance = calculate_euclidean_distance(datapoint, model_point)
            #print(f'The distance between {model_point} and {datapoint} is {distance}')
            if minimum_distance is None or minimum_distance > distance:
                minimum_distance = distance
                closest_point = model_point
        correspondences[:, index] = np.transpose(closest_point)

    sse = 1.1  # squared sum error - initial arbitrary value
    iteration = 0
    while sse > 0.1 and iteration < max_iterations:  # maximum_iterations is a safeguard to prevent infinite loop. Should be removed later

        if save_figure or show_plot:
            if two_dimension:
                plot_utils.plot_2d_model_and_data(model_view, data_view,
                                                  figure_name=f'{os.path.join(PLOTS_DIR, "iteration"+str(iteration)+".png" )}',
                                                  dpi=300, title=f'iteration {iteration}', save_figure=save_figure)
            else:
                plot_utils.plot_3d_model_and_data(model_view, data_view,
                                                  figure_name=f'{os.path.join(PLOTS_DIR, "iteration"+str(iteration)+".png" )}',
                                                  title=f'iteration {iteration}', save_figure=save_figure, dpi=300)

        #print(f'Find sse  for {altered_data} and {model} ')
        sse = error_icp(data_view, model_view)
        print(f"The sse at iteration {iteration}  is {sse}")

        # print(f'The correspondences are {correspondences}')

        cross_cov_mat = calculate_cross_covariance_matrix(data_view, correspondences)
        u, s, v_transpose = np.linalg.svd(cross_cov_mat)
        v = np.transpose(v_transpose)
        u_transpose = np.transpose(u)
        det_u_dot_det_v = np.linalg.det(u) * np.linalg.det(v)
        # print(f'The detu.detv is {det_u_dot_det_v}')
        if round(det_u_dot_det_v) == 1:
            s = np.diag([1, 1, 1])
        elif round(det_u_dot_det_v) == -1:
            s = np.diag([1, 1, -1])
        # r = v @ u_transpose # eqn 7
        rotation_matrix = v @ s @ u_transpose  # eqn 8
        translation_matrix = model_view_mean - np.matmul(rotation_matrix, data_view_mean) # eqn 9

        ultimate_rotational_matrix += rotation_matrix
        ultimate_translational_matrix += translation_matrix


        print(f'The rotation and translation for iteration {iteration} is {rotation_matrix} and {translation_matrix} respectively')

        #print(f'The rotational matrix is {r} and translation is {t}')
        transformed_matrix = apply_transformation(data_view, rotation_matrix, translation_matrix)

        #print(f'{model[0, ]} and {model[1, ]}')
        data_view = transformed_matrix
        data_view_mean = calculate_centroid(data_view)

        # calculate the correspondence using ICP
        for index, datapoint in enumerate(np.transpose(data_view)):
            closest_point = None
            minimum_ssm = None
            datapoint = datapoint.reshape((3, 1))
            #print(f'To find the distance for datapoint {datapoint}')
            for model_point in np.transpose(model_view):
                model_point = model_point.reshape((3, 1))
                error = (np.matmul(rotation_matrix, datapoint) + translation_matrix) - model_point
                ssm = float(error[0]**2 + error[1]**2 + error[2]**2)
                #print(f'The ssm for {model_point} and {datapoint} is {ssm}')
                if minimum_ssm is None or minimum_ssm > ssm:
                    minimum_ssm = ssm
                    closest_point = model_point
            correspondences[:, index] = np.transpose(closest_point)
        iteration += 1
    return ultimate_rotational_matrix, ultimate_translational_matrix

