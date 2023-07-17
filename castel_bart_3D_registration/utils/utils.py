# Author : Roche Christopher
# 23/05/23 - 16:34:27
import numpy
import numpy as np
import math
import os

PLOTS_DIR = '../plots'


def create_rectangle_points(length, width, start_at=(1, 1)):
    """
    :param length: Length of the rectangle
    :param width: Width of the rectangle
    :param start_at: The starting point of the rectangle; left bottom point of the rectangle
    :return: a set of 3-D points which define the rectangle with z=0
    """
    matrix = np.empty((0, 3), int)
    xinit = start_at[0]
    yinit = start_at[1]
    # create points for left bottom to top
    for i in range(yinit, yinit + width):
        matrix = np.append(matrix, [[xinit, i, 0]], axis=0)
    for i in range(xinit, xinit+length):
        matrix = np.append(matrix, [[i, yinit+width, 0]], axis=0)
    for i in range(yinit+width, yinit, -1):
        matrix = np.append(matrix, [[xinit+length, i, 0]], axis=0)
    for i in range(xinit+length, xinit, -1):
        matrix = np.append(matrix, [[i, yinit, 0]], axis=0)

    return matrix.transpose()


def rotate_matrix(matrix, degrees, axis='z', inverse=False):
    """"
    :param matrix: This parameter takes the matrix that has to be rotated
    :param degrees: The degrees to which the matrix has to be rotated. postive: counterclockwise, negative: clockwise
    :return: rotated matrix
    """

    if axis.lower() == 'z':
        rotational_matrix = np.array([
            [math.cos(math.radians(degrees)), -math.sin(math.radians(degrees)), 0],
            [math.sin(math.radians(degrees)), math.cos(math.radians(degrees)), 0],
            [0, 0, 1]
        ])
    elif axis.lower() == 'y':
        rotational_matrix = np.array([
            [math.cos(math.radians(degrees)), 0, -math.sin(math.radians(degrees))],
            [0, 1, 0],
            [math.sin(math.radians(degrees)), 0, math.cos(math.radians(degrees))]
        ])
    else:
        # rotate the matrix with respect to 'x'
        rotational_matrix = np.array([
            [1, 0, 0],
            [0, math.cos(math.radians(degrees)), -math.sin(math.radians(degrees))],
            [0, math.sin(math.radians(degrees)), math.cos(math.radians(degrees))]
        ])
    if inverse:
        rotational_matrix = rotational_matrix.transpose()
    rotated_matrix = np.matmul(rotational_matrix, matrix)
    return rotated_matrix


def translate_matrix(matrix, x, y, z):
    """
    :param matrix: The matrix that has to be translated; ,matrix shape would be 3 X number_of_points
    :param x: The magnitude to which the matrix has to be moved in x axis
    :param y: The magnitude to which the matrix has to be moved in y axis
    :param z: The magnitude to which the matrix has to be moved in z axis
    :return: translated matrix
    """
    # the following was implemented based on the matrix translation from google.
    translational_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    matrix = np.append(matrix, np.ones((1, matrix.shape[1])), axis=0)
    translated_matrix = np.matmul(translational_matrix, matrix)
    translated_matrix = translated_matrix[0:3,:]
    return translated_matrix


def calculate_centroid(matrix):
    """
    This method returns the mean of the given matrix in the shape 1x3
    """
    #print(f"The matrix for which centroid has to be found is {matrix}")
    matrix_shape = matrix.shape
    summation = np.zeros((matrix_shape[0], 1))
    for i in range(matrix_shape[0]):
        summation[i, 0] = sum(matrix[i, :])
    #print(f"The summation of the above matrix points is {summation}")
    centroid = summation/matrix_shape[1]

    return centroid


def calculate_q(matrix, centroid):

    q = matrix - centroid
    return q


def find_rotation(q_matrix, q_prime_matrix):
    summation_numerator = 0
    summation_denominator = 0
    for i in range(q_matrix.shape[1]):
        summation_numerator += q_matrix[0, i] * q_prime_matrix[1, i] - q_prime_matrix[0, i] * q_matrix[1, i]
        summation_denominator += q_matrix[0, i] * q_prime_matrix[0, i] + q_matrix[1, i] * q_prime_matrix[1, i]

    sigma1 = math.atan(summation_numerator/summation_denominator)
    print(f"The sigma1 is {sigma1}")

    r = get_r_vector(q_matrix)
    rho = get_r_vector(q_prime_matrix)

    print(f"The r vector is {r}")

    alpha = get_angle_from_polar(r, q_matrix)
    beta = get_angle_from_polar(rho, q_prime_matrix)
    print(f"Angle alpha is {alpha}")

    print("The following is the test for polar functions. It should list the y points of q consequtively")
    for i in range(r.size):
        print(r[i]*math.sin(alpha[i]))

    summation_numerator = 0
    summation_denominator = 0
    for i in range(q_matrix.shape[1]):
        summation_numerator += rho[i] * r[i] * math.sin(beta[i] - alpha[i])
        summation_denominator += rho[i] * r[i] * math.cos(beta[i]*alpha[i])

    sigma2 = math.atan(summation_numerator/summation_denominator)
    print(f"The sigma2 is {sigma2}")

    ls_sigma1 = get_square_error_for_angle(sigma1, r, rho, alpha, beta)
    ls_sigma2 = get_square_error_for_angle(sigma2, r, rho, alpha, beta)

    print(f"Least squares for sigma1 and sigma2 are {ls_sigma1} and {ls_sigma2} respectively")

def find_translation(rotation_matrix, centroid_p, centroid_p_prime):
    return centroid_p_prime - np.matmul(rotation_matrix, centroid_p)


def get_r_vector(q):
    r = np.zeros(q.shape[1])
    for i in range(q.shape[1]):
        r[i] = math.sqrt(q[0, i]**2 + q[1, i]**2)
    return r


def get_angle_from_polar(r, q):
    angle = np.zeros(q.shape[1])
    for i in range(q.shape[1]):
        angle[i] = math.acos(q[0,  i]/r[i])
    return angle


def get_square_error_for_angle(angle, r_vector, rho_vector, alpha, beta):
    summation = 0
    e = 2.71828
    for i in range(r_vector.size):
        summation += ((rho_vector[i]*e**(1j*beta[i])) - r_vector[i] * e**(1j * (alpha[i] + angle))) ** 2
    return math.sqrt(summation)


def calculate_euclidean_distance(point1, point2):
    #print(f'calculate_euclidean between {point1} and {point2}')
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)


def calculate_cross_covariance_matrix(data, correspondences, model_mean=None):
    if model_mean is None:
        model_mean = calculate_centroid(correspondences)
    data_mean = calculate_centroid(data)
    meaned_data = data - data_mean
    meaned_correspondences = correspondences - model_mean

    cross_cov_mat = numpy.matmul(meaned_data, np.transpose(meaned_correspondences))
    print(f'Cross covariance matrix is {cross_cov_mat}')

    return cross_cov_mat


def apply_transformation(matrix, rotation, translation):
    return np.matmul(rotation, matrix) + translation



