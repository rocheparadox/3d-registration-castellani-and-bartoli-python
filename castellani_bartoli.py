# Author : Roche Christopher
# 19/06/23 - 17:16:12

# The following code contains implementation of the work on 3d registration by Umberto Castellani and Adrien Bartoli
# http://encov.ip.uca.fr/publications/pubfiles/2020_Castellani_etal_3DIAA_registration.pdf

import math
import numpy as np
from matplotlib import pyplot
from utils import *
from animate_plots import animate_plots

max_iterations = 50
use_arun = False


def error_icp(altered_data, correspondences):
    sse = 0 # Squared sum of error
    for index, data in enumerate(altered_data):
        sse += calculate_euclidean_distance(data, correspondences[index])

    return sse


if __name__ == '__main__':

    # empty plots directory
    print('Cleaning plots dir')
    clean_plots_dir()
    print('Done cleaning plots directory')

    # Create a 3-D point set
    model = np.transpose(np.array([[1, 1, 0], [3, 3, 0], [5, 5, 0], [7, 6, 0], [8, 5, 0], [9, 4, 0], [11, 5, 0],
                                   [13, 7, 0], [15, 12, 0], [16, 14, 0], [18, 13, 0], [20, 13, 0], [21, 7, 0],
                                   [23, 5, 0], [24, 3, 0], [25, 1, 0], [26, 0, 0], [28, -3, 0],  [29, -4, 0],
                                   [30, -7, 0], [31, -8, 0], [33, -10, 0], [34, -9, 0], [35, -11, 0],  [37, -13, 0],
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
    # Calculate centroids
    model_mean = calculate_centroid(model)
    altered_data_mean = calculate_centroid(altered_data)
    print(f'The means of model and data are {model_mean} and {altered_data_mean} respectively')

    # Move the centroid of data view to the centroid of model view
    # mean_difference = altered_data_mean - model_mean
    # print(f'Altered data is {altered_data}')
    # altered_data = altered_data - mean_difference
    # print(f'Altered data after moving the centroid is {altered_data}')

    correspondences = np.empty(model.shape)
    # Find correspondence points using the euclidean distance
    for index, datapoint in enumerate(np.transpose(altered_data)):
        closest_point = None
        minimum_distance = None
        #print(f'To find the distance for datapoint {datapoint}')
        for model_point in np.transpose(model):
            distance = calculate_euclidean_distance(datapoint, model_point)
            #print(f'The distance between {model_point} and {datapoint} is {distance}')
            if minimum_distance is None or minimum_distance > distance:
                minimum_distance = distance
                closest_point = model_point
        correspondences[:, index] = np.transpose(closest_point)

    sse = 1.1  # squared sum error
    iteration = 0
    while sse > 0.1 and iteration < max_iterations:  # iteration < 10 is a safeguard to prevent infinite loop. Should be removed later

        pyplot.axis([-20, 50, -40, 40])
        pyplot.scatter(model[0, ], model[1, ])
        pyplot.plot(model[0, ], model[1, ])

        pyplot.scatter(altered_data[0, ], altered_data[1, ])
        pyplot.plot(altered_data[0, ], altered_data[1, ])
        pyplot.title(f'iteration {iteration}')
        print(f'saving image{iteration}')
        pyplot.savefig(f'plots/iteration{iteration}.png', dpi=300)
        pyplot.clf()  # clear the plot

        #print(f'Find sse  for {altered_data} and {model} ')
        sse = error_icp(altered_data, model)
        print(f"The sse at iteration {iteration}  is {sse}")

        # print(f'The correspondences are {correspondences}')

        cross_cov_mat = calculate_cross_covariance_matrix(altered_data, correspondences)
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
        r = v @ s @ u_transpose  # eqn 8

        t = model_mean - np.matmul(r, altered_data_mean)
        print(f'The rotation and translation for iteration {iteration} is {r} and {t} respectively')

        #print(f'The rotational matrix is {r} and translation is {t}')
        transformed_matrix = apply_transformation(altered_data, r, t)

        #print(f'{model[0, ]} and {model[1, ]}')
        altered_data = transformed_matrix
        altered_data_mean = calculate_centroid(altered_data)

        # calculate the correspondence using ICP
        for index, datapoint in enumerate(np.transpose(altered_data)):
            closest_point = None
            minimum_ssm = None
            datapoint = datapoint.reshape((3, 1))
            #print(f'To find the distance for datapoint {datapoint}')
            for model_point in np.transpose(model):
                model_point = model_point.reshape((3, 1))
                error = (np.matmul(r, datapoint) + t) - model_point
                ssm = float(error[0]**2 + error[1]**2 + error[2]**2)
                #print(f'The ssm for {model_point} and {datapoint} is {ssm}')
                if minimum_ssm is None or minimum_ssm > ssm:
                    minimum_ssm = ssm
                    closest_point = model_point
            correspondences[:, index] = np.transpose(closest_point)
        iteration += 1

    # animate the plot images
    animate_plots()
    exit()

    #print(f"The centroid of p is {datapoint_mean}")
    #Calculate q and q'


    # use iterative algorithm to find the angles of rotation
    #rotation = find_rotation(q, q_prime)
    # Find the translation
    #translation = find_translation(rotation, centroid_p, centroid_p_prime)