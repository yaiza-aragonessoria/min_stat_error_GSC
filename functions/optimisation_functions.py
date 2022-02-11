import numpy as np
import scipy as sc
from scipy.stats import chi2
import scipy.stats as stats
from scipy.stats import unitary_group
import time
import math as mth
import matplotlib.pyplot as plt

import functions.basic_functions as bf
import functions.fake_data_functions as fd

np.set_printoptions(suppress=True)

# Matrix L wrt 25 angles
def L(vec_theta, F):
    theta1 = vec_theta[0]
    theta2 = vec_theta[1]
    theta3 = vec_theta[2]
    theta4 = vec_theta[3]
    theta5 = vec_theta[4]
    theta6 = vec_theta[5]
    theta7 = vec_theta[6]
    theta8 = vec_theta[7]
    theta9 = vec_theta[8]
    theta10 = vec_theta[9]
    theta11 = vec_theta[10]
    theta12 = vec_theta[11]
    theta13 = vec_theta[12]
    theta14 = vec_theta[13]
    theta15 = vec_theta[14]
    theta16 = vec_theta[15]
    theta17 = vec_theta[16]
    theta18 = vec_theta[17]
    theta19 = vec_theta[18]
    theta20 = vec_theta[19]
    theta21 = vec_theta[20]
    theta22 = vec_theta[21]
    theta23 = vec_theta[22]
    theta24 = vec_theta[23]
    theta25 = vec_theta[24]
    L = (F[0]+F[1]-1)*np.array([[0, 0, 0, 0, -2 * np.sin(theta15), 0, 0, 0, 0, 2 * np.sin(theta15), 0, 0, 0, 0, 0],
         [0, 0, 0, -2 * np.sin(theta1), 0, 0, -2 * np.sin(theta1), 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, -2 * np.sin(theta16), 0, 0, -2 * np.sin(theta16), 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, -2 * np.sin(theta2), 0, 0, -2 * np.sin(theta2), 0, 0, 0, 0],
         [-2 * np.sin(theta17), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * np.sin(theta17), 0, 0],
         [0, -2 * np.sin(theta18), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * np.sin(theta18), 0],
         [0, -2 * np.sin(theta19) * np.sin(theta3), 0, -2 * np.cos(theta19) * np.sin(theta3),
          -2 * np.cos(theta3) * np.sin(theta19), 0, -2 * np.cos(theta19) * np.sin(theta3), 0, 0, 2 * np.sin(theta19), 0,
          0,
          0, 0, 0],
         [0, 0, 0, -2 * np.cos(theta20) * np.sin(theta4), 0, -2 * np.sin(theta20),
          -2 * np.cos(theta20) * np.sin(theta4), 0,
          -2 * np.cos(theta4) * np.sin(theta20), 0, 0, 0, -2 * np.sin(theta20) * np.sin(theta4), 0, 0],
         [0, 0, 0, 0, -2 * np.cos(theta5) * np.sin(theta21), 0, 0, -2 * np.cos(theta21) * np.sin(theta5), 0,
          2 * np.sin(theta21), -2 * np.cos(theta21) * np.sin(theta5), 0, 2 * np.sin(theta21) * np.sin(theta5), 0, 0],
         [-2 * np.cos(theta6) * np.sin(theta7), 0, 0, -2 * np.cos(theta7) * np.sin(theta6), 0, 0, -2 * np.sin(theta6),
          0, 0,
          0, 0, 0, -2 * np.sin(theta7), 0, 0],
         [0, -2 * np.sin(theta22) * np.sin(theta8), 0, 0, 0, -2 * np.sin(theta22), 0,
          -2 * np.cos(theta22) * np.sin(theta8),
          -2 * np.cos(theta8) * np.sin(theta22), 0, -2 * np.cos(theta22) * np.sin(theta8), 0, 0, 0, 0],
         [0, -2 * np.cos(theta9) * np.sin(theta10), 0, 0, 0, 0, 0, -2 * np.cos(theta10) * np.sin(theta9), 0, 0,
          -2 * np.sin(theta9), 0, 0, -2 * np.sin(theta10), 0],
         [-2 * np.cos(theta11) * np.sin(theta23), -2 * np.cos(theta23) * np.sin(theta11),
          2 * np.sin(theta11) * np.sin(theta23), 0, 0, 0, 0, 0, 0, 0, 0, 0, -2 * np.cos(theta11) * np.sin(theta23),
          -2 * np.cos(theta23) * np.sin(theta11), 2 * np.sin(theta11) * np.sin(theta23)],
         [0, 0, -2 * np.sin(theta12) * np.sin(theta24), -2 * np.cos(theta24) * np.sin(theta12),
          -2 * np.cos(theta24) * np.sin(theta12), -2 * np.cos(theta12) * np.sin(theta24),
          -2 * np.cos(theta24) * np.sin(theta12), -2 * np.cos(theta12) * np.sin(theta24),
          -2 * np.cos(theta12) * np.sin(theta24), 2 * np.cos(theta24) * np.sin(theta12),
          -2 * np.cos(theta12) * np.sin(theta24), -4 * np.sin(theta12) * np.sin(theta24), 0, 0,
          -2 * np.sin(theta12) * np.sin(theta24)], [-2 * np.cos(theta13) * np.cos(theta14) * np.sin(theta25),
                                                    -2 * np.cos(theta13) * np.cos(theta25) * np.sin(theta14),
                                                    2 * np.cos(theta13) * np.sin(theta14) * np.sin(theta25),
                                                    -2 * np.cos(theta14) * np.cos(theta25) * np.sin(theta13), 0,
                                                    -2 * np.sin(theta13) * np.sin(theta25),
                                                    -2 * np.cos(theta25) * np.sin(theta13), 0, 0, 0, 0, 0,
                                                    -2 * np.cos(theta14) * np.sin(theta25),
                                                    -2 * np.cos(theta25) * np.sin(theta14),
                                                    2 * np.sin(theta14) * np.sin(theta25)]])
    return L

# Function to optimise the mean squared distance wrt 25 angles
def mean_distance2(vec_theta, N, F):

    matrix_L = L(vec_theta, F) # Constructing L matrix
    cov_matrix = cov_matrixR(vec_theta, N, F) # Constructing covariance matrix of responses

    if bf.is_invertible(np.array(matrix_L))==True: # If L is non-singular, the function computes the mean squared distance
        return np.real(np.matrix.trace(np.linalg.inv(matrix_L) @ cov_matrix @ np.matrix.transpose(np.linalg.inv(matrix_L))))
    else: # If L is singular, the function returns infinity.
        return mth.inf

# Function that will be given to the optimiser when imperfect measurements are considered
def mysuperfunc(vec_theta, grad):
    N = 10**6 # Number of runs of the "experiment in the lab"
    F = [0.99, 0.98] # Measurement fidelities. If F=[1,1], the measurements are perfect.

    return mean_distance2(vec_theta, N, F)

# Function that will be given to the optimiser when perfect measurements are considered
def mysuperfunc1(vec_theta, grad):
    N = 10**6 # Number of runs of the "experiment in the lab"
    F = [1, 1] # Measurement fidelities. If F=[1,1], the measurements are perfect.

    return mean_distance2(vec_theta, N, F)

# Covariance matrix of the error parameters, i.e., L^(-1) Sigma L^(-T) wrt 25 angles
def cov_matrixP(vec_theta, N, F):

    matrix_L = L(vec_theta, F) # Constructing L matrix
    cov_matrix_R = cov_matrixR(vec_theta, N, F) # Constructing covariance matrix of responses

    if bf.is_invertible(np.array(matrix_L))==True: # If L is non-singular, the function computes the covariance matrix of the error parameters
        return np.linalg.inv(matrix_L) @ cov_matrix_R @ np.matrix.transpose(np.linalg.inv(matrix_L))
    else: # If L is singular, the functions returns infinity.
        return mth.inf

# Covariance matrix of the responses wrt 25 angles
def cov_matrixR(vec_theta,N,F):
    # Assigning the angles
    theta1 = vec_theta[0]
    theta2 = vec_theta[1]
    theta3 = vec_theta[2]
    theta4 = vec_theta[3]
    theta5 = vec_theta[4]
    theta6 = vec_theta[5]
    theta7 = vec_theta[6]
    theta8 = vec_theta[7]
    theta9 = vec_theta[8]
    theta10 = vec_theta[9]
    theta11 = vec_theta[10]
    theta12 = vec_theta[11]
    theta13 = vec_theta[12]
    theta14 = vec_theta[13]
    theta15 = vec_theta[14]
    theta16 = vec_theta[15]
    theta17 = vec_theta[16]
    theta18 = vec_theta[17]
    theta19 = vec_theta[18]
    theta20 = vec_theta[19]
    theta21 = vec_theta[20]
    theta22 = vec_theta[21]
    theta23 = vec_theta[22]
    theta24 = vec_theta[23]
    theta25 = vec_theta[24]

    # Computing the covariance matrix of the responses
    cov_matrix_R= [[(1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta15)) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta1)) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta16)) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta2)) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta17)) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta18)) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta19) * np.cos(theta3)) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta20) * np.cos(theta4)) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta21) * np.cos(theta5)) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta6) * np.cos(theta7)) ** 2) / (4 * N), 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta22) * np.cos(theta8)) ** 2) / (4 * N), 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta10) * np.cos(theta9)) ** 2) / (4 * N), 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta11) * np.cos(theta23)) ** 2) / (4 * N), 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta12) * np.cos(theta24)) ** 2) / (4 * N), 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              (1 - (-F[1]+F[0]+(-1+F[1]+F[0])*np.cos(theta13) * np.cos(theta14) * np.cos(theta25)) ** 2) / (4 * N)]]

    return cov_matrix_R

# Covariance matrix of the responses wrt 1 angles
def cov_matrixR1(vec_theta,N):
    # Assigning the angle
    theta = vec_theta[0]

    # Computing the covariance matrix of the responses
    cov_matrix_R= [[(1 - np.cos(theta) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, (1 - np.cos(theta) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, (1 - np.cos(theta) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, (1 - np.cos(theta) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, (1 - np.cos(theta) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, (1 - np.cos(theta) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, (1 - np.cos(theta) ** 2 * np.cos(theta) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, (1 - np.cos(theta) ** 2 * np.cos(theta) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, (1 - np.cos(theta) ** 2 * np.cos(theta) ** 2) / (4 * N), 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, (1 - np.cos(theta) ** 2 * np.cos(theta) ** 2) / (4 * N), 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1 - np.cos(theta) ** 2 * np.cos(theta) ** 2) / (4 * N), 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1 - np.cos(theta) ** 2 * np.cos(theta) ** 2) / (4 * N), 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1 - np.cos(theta) ** 2 * np.cos(theta) ** 2) / (4 * N), 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1 - np.cos(theta) ** 2 * np.cos(theta) ** 2) / (4 * N), 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              (1 - np.cos(theta) ** 2 * np.cos(theta) ** 2 * np.cos(theta) ** 2) / (4 * N)]]

    return cov_matrix_R