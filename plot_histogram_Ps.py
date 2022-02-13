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

F=[1, 1]
N=10**6

# READING DATA

J=10**4
## Reading estimated error parameters from optimised thetas
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_Ps.txt") as f:
    Ps_opt = [[num for num in line.split(' ')] for line in f]
    del Ps_opt[0]
    Ps_opt = [[float(item) for item in line] for line in Ps_opt]
f.close()

J=10**4
## Reading estimated error parameters from experimentalist GSC version
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_Ps.txt") as f:
    Ps = [[num for num in line.split(' ')] for line in f]
    del Ps[0]
    Ps = [[float(item) for item in line] for line in Ps]
f.close()

J=10**4
## Reading mean vector from optimised thetas
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_mean_vector.txt") as f:
    mean_vector_opt = [num for num in f]
    del mean_vector_opt[0:17]
    del mean_vector_opt[15:31]
    mean_vector_opt = [float(item) for item in mean_vector_opt]
f.close()

J=10**4
## Reading mean vector of experimentalist GSC version
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_mean_vector.txt") as f:
    mean_vector = [num for num in f]
    del mean_vector[0:17]
    del mean_vector[15:31]
    mean_vector = [float(item) for item in mean_vector]
f.close()

J=10**4
## Reading ps for optimal angles
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_pss.txt") as f:
    ps_opt = [num for num in f]
    del ps_opt[0]
    ps_opt = [float(item) for item in ps_opt]
f.close()

J=10**4
## Reading ps of experimentalist GSC version
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_pss.txt") as f:
    ps = [num for num in f]
    del ps[0]
    ps = [float(item) for item in ps]
f.close()

J=10**4
## Reading covariance matrix from optimised thetas
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_cov_matrix.txt") as f:
    cov_matrix_opt = [[num for num in line.split(' ')] for line in f]
    del cov_matrix_opt[0:17]
    del cov_matrix_opt[15:31]
    cov_matrix_opt = [[float(item) for item in line] for line in cov_matrix_opt]
f.close()

J=10**4
## Reading covariance matrix of experimentalist GSC version
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_cov_matrix.txt") as f:
    cov_matrix = [[num for num in line.split(' ')] for line in f]
    del cov_matrix[0:17]
    del cov_matrix[15:31]
    cov_matrix = [[float(item) for item in line] for line in cov_matrix]
f.close()


# HISTOGRAMS
## For each error parameter, Ps[k], we plot the following:
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
    var_list = np.var(Ps[k])
    mean_list = np.mean(Ps[k])

    ## Experimental parameters, i.e., from the data of the simulation with the optimal angles
    var_list_opt = np.var(Ps_opt[k])
    mean_list_opt = np.mean(Ps_opt[k])

    ## Theoretical parameters of the GSC version with all angles pi/2
    var_th = cov_matrix[k][k]
    mean_th = mean_vector[k]

    ## Theoretical parameters of the GSC version with the optimal angles
    var_th_opt = cov_matrix_opt[k][k]
    mean_th_opt = mean_vector_opt[k]

    ## Plotting the histogram with the data obtained simulating the calibration with the GSC version with all angles pi/2
    plt.subplot(2, 1, 1)
    plt.hist(Ps[k], bins='auto', color='g', edgecolor='k', stacked=True, density=True)
    plt.title("Error parameter " + " k = " + str(k) + " for F=" + str(F))
    plt.xlabel('P[k]')

    # Creating the gaussian distribution with experimental variance and mean for the GSC version with all angles pi/2
    x = np.linspace(mean_list - 4 * np.sqrt(var_list), mean_list + 4 * np.sqrt(var_list), 100)
    plt.plot(x, stats.norm.pdf(x, mean_list, np.sqrt(var_list)))

    # Creating the gaussian distribution with theoretical variance and mean for the GSC version with all angles pi/2
    x = np.linspace(mean_th - 4 * np.sqrt(var_th), mean_th + 4 * np.sqrt(var_th), 100)
    plt.plot(x, stats.norm.pdf(x, mean_th, np.sqrt(var_th)), color='orange', linestyle='-.')

    # Plotting the histogram with the data obtained simulating the calibration with the GSC version with the optimal angles
    plt.subplot(2, 1, 2)
    plt.hist(Ps_opt[k], bins='auto', color='g', edgecolor='k', stacked=True, density=True)
    plt.title("Error parameter " + " k = " + str(k) + " for F=" + str(F))
    plt.xlabel('P_opt[k]')

    # Creating the gaussian distribution with experimental variance and mean for the GSC version with the optimal angles
    x = np.linspace(mean_list_opt - 4 * np.sqrt(var_list_opt), mean_list_opt + 4 * np.sqrt(var_list_opt), 100)
    plt.plot(x, stats.norm.pdf(x, mean_list_opt, np.sqrt(var_list_opt)))

    # Creating the gaussian distribution with theoretical variance and mean for the GSC version with the optimal angles
    x = np.linspace(mean_th_opt - 4 * np.sqrt(var_th_opt), mean_th_opt + 4 * np.sqrt(var_th_opt), 100)
    plt.plot(x, stats.norm.pdf(x, mean_th_opt, np.sqrt(var_th_opt)), color='orange', linestyle='-.')

    print("numerical mean =", mean_list)
    print("theoretical mean =", mean_th)
    print("real error parameter =", ps[k])

    print("numerical var =", var_list)
    print("theoretical var =", var_th)

    print("   ")

    print("numerical mean for opt =", mean_list_opt)
    print("theoretical mean for opt =", mean_th_opt)
    print("real error parameter =", ps_opt[k])

    print("numerical var for opt =", var_list_opt)
    print("theoretical var for opt =", var_th_opt)


    plt.tight_layout()
    plt.show()

    k += 1


print("   ")
print("#------------- IMPERFECT MEASUREMENTS -------------#")
print("   ")
# Code analogous as the code for perfect measurements.

F=[0.99, 0.98]
N=10**6

# READING DATA

J=10**4
## Reading estimated error parameters from optimised thetas
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_Ps.txt") as f:
    Ps_opt = [[num for num in line.split(' ')] for line in f]
    del Ps_opt[0]
    Ps_opt = [[float(item) for item in line] for line in Ps_opt]
f.close()

J=10**4
## Reading estimated error parameters from experimentalist GSC version
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_Ps.txt") as f:
    Ps = [[num for num in line.split(' ')] for line in f]
    del Ps[0]
    Ps = [[float(item) for item in line] for line in Ps]
f.close()

J=10**4
## Reading mean vector from optimised thetas
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_mean_vector.txt") as f:
    mean_vector_opt = [num for num in f]
    del mean_vector_opt[0:17]
    del mean_vector_opt[15:31]
    mean_vector_opt = [float(item) for item in mean_vector_opt]
f.close()

J=10**4
## Reading mean vector of experimentalist GSC version
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_mean_vector.txt") as f:
    mean_vector = [num for num in f]
    del mean_vector[0:17]
    del mean_vector[15:31]
    mean_vector = [float(item) for item in mean_vector]
f.close()

J=10**4
## Reading real error parameters, i.e., ps, for optimal angles
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_pss.txt") as f:
    ps_opt = [num for num in f]
    del ps_opt[0]
    ps_opt = [float(item) for item in ps_opt]
f.close()

J=10**4
## Reading real error parameters, i.e., ps, of experimentalist GSC version
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_pss.txt") as f:
    ps = [num for num in f]
    del ps[0]
    ps = [float(item) for item in ps]
f.close()

J=10**4
## Reading covariance matrix from optimised thetas
with open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_cov_matrix.txt") as f:
    cov_matrix_opt = [[num for num in line.split(' ')] for line in f]
    del cov_matrix_opt[0:17]
    del cov_matrix_opt[15:31]
    cov_matrix_opt = [[float(item) for item in line] for line in cov_matrix_opt]
f.close()

J=10**4
## Reading covariance matrix of experimentalist GSC version
with open("data/exp/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_cov_matrix.txt") as f:
    cov_matrix = [[num for num in line.split(' ')] for line in f]
    del cov_matrix[0:17]
    del cov_matrix[15:31]
    cov_matrix = [[float(item) for item in line] for line in cov_matrix]
f.close()

k = 0
while k < 15:
    print("     ")
    print("k =", k)

    var_list = np.var(Ps[k])
    mean_list = np.mean(Ps[k])

    var_list_opt = np.var(Ps_opt[k])
    mean_list_opt = np.mean(Ps_opt[k])

    var_th = cov_matrix[k][k]
    mean_th = mean_vector[k]

    var_th_opt = cov_matrix_opt[k][k]
    mean_th_opt = mean_vector_opt[k]

    plt.subplot(2, 1, 1)
    plt.hist(Ps[k], bins='auto', color='g', edgecolor='k', stacked=True, density=True)
    plt.title("Error parameter " + " k = " + str(k) + " for F=" + str(F))
    plt.xlabel('P[k]')

    x = np.linspace(mean_list - 4 * np.sqrt(var_list), mean_list + 4 * np.sqrt(var_list), 100)
    plt.plot(x, stats.norm.pdf(x, mean_list, np.sqrt(var_list)))

    x = np.linspace(mean_th - 4 * np.sqrt(var_th), mean_th + 4 * np.sqrt(var_th), 100)
    plt.plot(x, stats.norm.pdf(x, mean_th, np.sqrt(var_th)), color='orange', linestyle='-.')

    plt.subplot(2, 1, 2)
    plt.hist(Ps_opt[k], bins='auto', color='g', edgecolor='k', stacked=True, density=True)
    plt.title("Error parameter " + " k = " + str(k) + " for F=" + str(F))
    plt.xlabel('P_opt[k]')

    x = np.linspace(mean_list_opt - 4 * np.sqrt(var_list_opt), mean_list_opt + 4 * np.sqrt(var_list_opt), 100)
    plt.plot(x, stats.norm.pdf(x, mean_list_opt, np.sqrt(var_list_opt)))

    x = np.linspace(mean_th_opt - 4 * np.sqrt(var_th_opt), mean_th_opt + 4 * np.sqrt(var_th_opt), 100)
    plt.plot(x, stats.norm.pdf(x, mean_th_opt, np.sqrt(var_th_opt)), color='orange', linestyle='-.')

    print("numerical mean =", mean_list)
    print("theoretical mean =", mean_th)
    print("real error parameter =", ps[k])

    print("numerical var =", var_list)
    print("theoretical var =", var_th)

    print("   ")

    print("numerical mean for opt =", mean_list_opt)
    print("theoretical mean for opt =", mean_th_opt)
    print("real error parameter =", ps_opt[k])

    print("numerical var for opt =", var_list_opt)
    print("theoretical var for opt =", var_th_opt)


    plt.tight_layout()
    plt.show()

    k += 1