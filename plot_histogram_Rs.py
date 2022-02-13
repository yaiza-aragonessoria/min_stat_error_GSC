# This file plots histograms of the errors parameters of data obtained from simulations made with fake_data.py

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'C:/Users/Yaiza/PycharmProjects/min_stat_error_GSC/functions/')
import basic_functions as bf
import fake_data_functions as fd
import optimisation_functions as opt


np.set_printoptions(formatter={'float_kind':'{:.4e}'.format}) #Setting the number of decimals to be printed

print("#------------- PERFECT MEASUREMENTS -------------#")
print("    ")
F=[1,1]
N=10**6

# READING DATA

## Reading responses from optimised thetas
J=10**4
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_Rs.txt") as f:
    Rs_opt = [[num for num in line.split(' ')] for line in f]
    del Rs_opt[0]
    Rs_opt = [[float(item) for item in line] for line in Rs_opt]
f.close()
# print("Rs =", Rs)

## Reading responses of experimentalist GSC version
J=10**4
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_Rs.txt") as f:
    Rs = [[num for num in line.split(' ')] for line in f]
    del Rs[0]
    Rs = [[float(item) for item in line] for line in Rs]
f.close()

## Reading mean vector from optimised thetas
J=10**4
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_mean_vector.txt") as f:
    mean_vector_opt = [num for num in f]
    del mean_vector_opt[0]
    del mean_vector_opt[15:47]
    mean_vector_opt = [float(item) for item in mean_vector_opt]
f.close()

## Reading mean vector of experimentalist GSC version
J=10**4
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_mean_vector.txt") as f:
    mean_vector = [num for num in f]
    del mean_vector[0]
    del mean_vector[15:47]
    mean_vector = [float(item) for item in mean_vector]
f.close()

## Reading covariance matrix from optimised thetas
J=10**4
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_cov_matrix.txt") as f:
    cov_matrix_opt = [[num for num in line.split(' ')] for line in f]
    del cov_matrix_opt[0]
    del cov_matrix_opt[15:47]
    cov_matrix_opt = [[float(item) for item in line] for line in cov_matrix_opt]
f.close()

## Reading covariance matrix of experimentalist GSC version
J=10**4
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_cov_matrix.txt") as f:
    cov_matrix = [[num for num in line.split(' ')] for line in f]
    del cov_matrix[0]
    del cov_matrix[15:47]
    cov_matrix = [[float(item) for item in line] for line in cov_matrix]
f.close()


# HISTOGRAMS
## For each response, Rs[k], we plot the following:
### a histogram with the data obtained simulating the calibration with the GSC version with all angles pi/2
### a gaussian distribution with variance and mean computed using the data of the histogram of all angles pi/2
### a gaussian distribution with the theoretical variance and mean of the GSC version with all angles pi/2
### a histogram with the data obtained simulating the calibration with the GSC version with the optimal angles
### a gaussian distribution with variance and mean computed using the data of the histogram of the optimal angles
### a gaussian distribution with the theoretical variance and mean of the GSC version with the optimal angles

k = 0
while k < 15:
    print("     ")
    print("k =", k)

    ## Experimental parameters, i.e., from the data of the simulation with all angles pi/2
    var_list = np.var(Rs[k])
    mean_list = np.mean(Rs[k])

    ## Experimental parameters, i.e., from the data of the simulation with the optimal angles
    var_list_opt = np.var(Rs_opt[k])
    mean_list_opt = np.mean(Rs_opt[k])

    ## Theoretical parameters of the GSC version with all angles pi/2
    var_th = cov_matrix[k][k]
    mean_th = mean_vector[k]

    ## Theoretical parameters of the GSC version with the optimal angles
    var_th_opt = cov_matrix_opt[k][k]
    mean_th_opt = mean_vector_opt[k]

    ## Plotting the histogram with the data obtained simulating the calibration with the GSC version with all angles pi/2
    plt.subplot(2, 1, 1)
    plt.hist(Rs[k], bins='auto', color='g', edgecolor='k', stacked=True, density=True)
    plt.title("Response " + " k = " + str(k) + " for F=" + str(F))
    plt.xlabel('R[k]')

    # Creating the gaussian distribution with experimental variance and mean for the GSC version with all angles pi/2
    x = np.linspace(mean_list - 4 * np.sqrt(var_list), mean_list + 4 * np.sqrt(var_list), 100)
    plt.plot(x, stats.norm.pdf(x, mean_list, np.sqrt(var_list)))

    # Creating the gaussian distribution with theoretical variance and mean for the GSC version with all angles pi/2
    x = np.linspace(mean_th - 4 * np.sqrt(var_th), mean_th + 4 * np.sqrt(var_th), 100)
    plt.plot(x, stats.norm.pdf(x, mean_th, np.sqrt(var_th)), color='orange', linestyle='-.')

    # Plotting the histogram with the data obtained simulating the calibration with the GSC version with the optimal angles
    plt.subplot(2, 1, 2)
    plt.hist(Rs_opt[k], bins='auto', color='g', edgecolor='k', stacked=True, density=True)
    plt.title("Response " + " k = " + str(k) + " for F=" + str(F))
    plt.xlabel('R_opt[k]')

    # Creating the gaussian distribution with experimental variance and mean for the GSC version with the optimal angles
    x = np.linspace(mean_list_opt - 4 * np.sqrt(var_list_opt), mean_list_opt + 4 * np.sqrt(var_list_opt), 100)
    plt.plot(x, stats.norm.pdf(x, mean_list_opt, np.sqrt(var_list_opt)))

    # Creating the gaussian distribution with theoretical variance and mean for the GSC version with the optimal angles
    x = np.linspace(mean_th_opt - 4 * np.sqrt(var_th_opt), mean_th_opt + 4 * np.sqrt(var_th_opt), 100)
    plt.plot(x, stats.norm.pdf(x, mean_th_opt, np.sqrt(var_th_opt)), color='orange', linestyle='-.')

    print("numerical mean =", mean_list)
    print("theoretical mean =", mean_th)

    print("numerical var =", var_list)
    print("theoretical var =", var_th)

    print("   ")

    print("numerical mean for opt =", mean_list_opt)
    print("theoretical mean for opt =", mean_th_opt)

    print("numerical var for opt =", var_list_opt)
    print("theoretical var for opt =", var_th_opt)

    plt.tight_layout()
    plt.show()

    k += 1


print("   ")
print("   ")
print("#------------- IMPERFECT MEASUREMENTS -------------#")
print("   ")
# Code analogous as the code for perfect measurements.

F=[0.99, 0.98]
N=10**6

# READING DATA

## Reading responses from optimised thetas
J=10**4
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_Rs.txt") as f:
    Rs_opt = [[num for num in line.split(' ')] for line in f]
    # print(differences)
    del Rs_opt[0]
    Rs_opt = [[float(item) for item in line] for line in Rs_opt]
f.close()
# print("Rs =", Rs)

## Reading responses of experimentalist GSC version
J=10**4
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_Rs.txt") as f:
    Rs = [[num for num in line.split(' ')] for line in f]
    del Rs[0]
    Rs = [[float(item) for item in line] for line in Rs]
f.close()

## Reading mean vector from optimised thetas
J=10**4
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_mean_vector.txt") as f:
    mean_vector_opt = [num for num in f]
    del mean_vector_opt[0]
    del mean_vector_opt[15:47]
    mean_vector_opt = [float(item) for item in mean_vector_opt]
f.close()

## Reading mean vector of experimentalist GSC version
J=10**4
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_mean_vector.txt") as f:
    mean_vector = [num for num in f]
    del mean_vector[0]
    del mean_vector[15:47]
    mean_vector = [float(item) for item in mean_vector]
f.close()

## Reading covariance matrix from optimised thetas
J=10**4
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_cov_matrix.txt") as f:
    cov_matrix_opt = [[num for num in line.split(' ')] for line in f]
    del cov_matrix_opt[0]
    del cov_matrix_opt[15:47]
    cov_matrix_opt = [[float(item) for item in line] for line in cov_matrix_opt]
f.close()

## Reading covariance matrix of experimentalist GSC version
J=10**4
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_cov_matrix.txt") as f:
    cov_matrix = [[num for num in line.split(' ')] for line in f]
    del cov_matrix[0]
    del cov_matrix[15:47]
    cov_matrix = [[float(item) for item in line] for line in cov_matrix]
f.close()

k = 0
while k < 15:
    print("     ")
    print("k =", k)

    var_list = np.var(Rs[k])
    mean_list = np.mean(Rs[k])

    var_list_opt = np.var(Rs_opt[k])
    mean_list_opt = np.mean(Rs_opt[k])

    var_th = cov_matrix[k][k]
    mean_th = mean_vector[k]

    var_th_opt = cov_matrix_opt[k][k]
    mean_th_opt = mean_vector_opt[k]

    plt.subplot(2, 1, 1)
    plt.hist(Rs[k], bins='auto', color='g', edgecolor='k', stacked=True, density=True)
    plt.title("Response " + " k = " + str(k) + " for F=" + str(F))
    plt.xlabel('R[k]')

    x = np.linspace(mean_list - 4 * np.sqrt(var_list), mean_list + 4 * np.sqrt(var_list), 100)
    plt.plot(x, stats.norm.pdf(x, mean_list, np.sqrt(var_list)))

    x = np.linspace(mean_th - 4 * np.sqrt(var_th), mean_th + 4 * np.sqrt(var_th), 100)
    plt.plot(x, stats.norm.pdf(x, mean_th, np.sqrt(var_th)), color='orange', linestyle='-.')

    plt.subplot(2, 1, 2)
    plt.hist(Rs_opt[k], bins='auto', color='g', edgecolor='k', stacked=True, density=True)
    plt.title("Response " + " k = " + str(k) + " for F=" + str(F))
    plt.xlabel('R_opt[k]')

    x = np.linspace(mean_list_opt - 4 * np.sqrt(var_list_opt), mean_list_opt + 4 * np.sqrt(var_list_opt), 100)
    plt.plot(x, stats.norm.pdf(x, mean_list_opt, np.sqrt(var_list_opt)))

    x = np.linspace(mean_th_opt - 4 * np.sqrt(var_th_opt), mean_th_opt + 4 * np.sqrt(var_th_opt), 100)
    plt.plot(x, stats.norm.pdf(x, mean_th_opt, np.sqrt(var_th_opt)), color='orange', linestyle='-.')

    print("numerical mean =", mean_list)
    print("theoretical mean =", mean_th)

    print("numerical var =", var_list)
    print("theoretical var =", var_th)

    print("   ")

    print("numerical mean for opt =", mean_list_opt)
    print("theoretical mean for opt =", mean_th_opt)

    print("numerical var for opt =", var_list_opt)
    print("theoretical var for opt =", var_th_opt)

    plt.tight_layout()
    plt.show()

    k += 1