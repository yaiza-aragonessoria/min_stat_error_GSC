# This project simulates the calibration of a perturbed CNOT with GSC using different angles for the single-qubit rotations.
# Then, it plots statistical error as measured by the mean squared distance as a function of the angle of the single-qubit rotation.

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


import functions.basic_functions as bf
import functions.fake_data_functions as fd
import functions.optimisation_functions as opt


# This command suppresses printing of small floats
np.set_printoptions(suppress=True)

# This command sets the printing options
np.set_printoptions(formatter={'float_kind':'{:.3e}'.format})

## GETTING MEAN SQUARED DISTANCE [i.e., tr(L^(-1)*Sigma*L^(-T))] IN TERMS OF ONE THETA
# """
M30 = np.kron(bf.pauli_z, np.identity(2))
M03 = np.kron(np.identity(2), bf.pauli_z)

M1 = M30
M2 = M30
M3 = M30
M4 = M30
M5 = M03
M6 = M03
M7 = M30
M8 = M30
M9 = M30
M10 = M03
M11 = M30
M12 = M03
M13 = M03
M14 = M30
M15 = M03
M16 = M30
M17 = M03

measurements = [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15]
in_st = [np.kron(np.kron(bf.up, bf.up), bf.dagga(np.kron(bf.up, bf.up)))]

N = 10 ** (6)  # Number of runs of the "experiment in the lab"
R = 10 ** 3  # Number of values of theta
delta = 2 * np.pi / R
F = [1, 1]

# CREATING THE PERTURBED CNOT and COMPUTING ps, i.e., the theoretical error parameters from

## If we want the CÑOT to be random...
# random coefficients of noise
# c = np.random.random(15) * 10 ** (-3)
# c = np.zeros(15)
# print("c=",c)

## If we've fixed the CÑOT...
# Reading c to fix CÑOT
with open("data/c_fixed.txt") as f:
    c = [num for num in f]
    del c[0]
    c = [float(item) for item in c]
f.close()

## Building HCNOT
HCNOT = -1j * sc.linalg.logm(bf.CNOT(1))
# print("HCNOT=", HCNOT)

## Building HCÑOT, i.e., introducing noise to HCNOT to build a CÑOT=exp(1j*HCÑOT)
HCÑOT = HCNOT
k = 0
while k in range(15):
    HCÑOT = HCÑOT + c[k] * bf.paulis[k]
    k += 1
# print("HCÑOT=",HCÑOT)

## Build a CÑOT, i.e., perturbed CNOT
CÑOT = sc.linalg.expm(1j * HCÑOT)
# print("CÑOT=",CÑOT)

## Computing the theoretical error parameters
k = 0
ps = np.zeros(15)
while k in range(15):
    ps[k] = -np.imag(np.matrix.trace(bf.CNOT(1) @ CÑOT @ bf.paulis[k])) / 4
    # print("ps[k] =", ps[k])
    k += 1

list_theta = []
list_mean_distance2 = []
list_cond_num = []

for r in range(R):
    if r % 10 ** 2 == 0:
        print("r =", r)

    theta = r * delta

    s1 = [["CNOT", 1], ["Rx", 1, theta]]
    s2 = [["Rx", 1, theta], ["CNOT", 1]]
    s3 = [["CNOT", 1], ["Ry", 1, theta]]
    s4 = [["Ry", 1, theta], ["CNOT", 1]]
    s5 = [["CNOT", 1], ["Rx", 2, theta]]
    s6 = [["CNOT", 1], ["Ry", 2, theta]]
    s7 = [["Rx", 1, theta], ["CNOT", 1], ["Rx", 1, theta]]
    s8 = [["Rx", 1, theta], ["CNOT", 1], ["Ry", 1, theta]]
    s9 = [["Ry", 1, theta], ["CNOT", 1], ["Rx", 1, theta]]
    s10 = [["Rx", 1, theta], ["Rx", 2, theta], ["CNOT", 1]]
    s11 = [["Ry", 1, theta], ["CNOT", 1], ["Ry", 1, theta]]
    s12 = [["Ry", 1, theta], ["Ry", 2, theta], ["CNOT", 1]]
    s13 = [["Ry", 2, theta], ["CNOT", 1], ["Rx", 2, theta]]
    s14 = [["Rx", 1, theta], ["CNOT", 1], ["CNOT", 1], ["Ry", 1, theta]]
    s15 = [["Rx", 1, theta], ["Ry", 2, theta], ["CNOT", 1], ["Rx", 2, theta]]

    sequences = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15]

    ## As the CNOT is the gate we want to calibrate, we change the sequences  such that the code applies a perturbed CNOT, i.e., a CÑOT, and not a CNOT.
    s = 0
    while s < len(sequences):
        g = 0
        while g < len(sequences[s]):
            if sequences[s][g][0] == "CNOT":
                sequences[s][g] = ["ex", HCÑOT]
            g += 1
        s += 1

    ## Computing the theoretical responses, i.e., R0, which have systematic errors, but no statistical error
    s = 0
    list_R0 = []
    while s < len(sequences):
        # print("s=", s + 1)
        after_gates = fd.gate_conc(sequences[s],
                                   in_st)  # Computing the state after applying the gates of sequence s
        # print("after_gates",s+1,"=", after_gates)

        R0 = fd.response(measurements[s], [after_gates], F)[
            0]  # Response of the measurement measurements[s] on the state after_gate.
        # print("R0[",s+1,"] =", R0)

        list_R0.append(R0)  # Adding the response to the list of responses.

        s += 1

    ## CONSRTUCTING MATRIX L
    L = (F[0] + F[1] - 1) * fd.constructL(sequences, measurements, in_st)
    # print("L=", L)
    # print("Condition number of L =", np.linalg.cond(L))

    if bf.is_invertible(L) == False:
        print("L is not invertible.")
        print("Value of theta =", theta, "skipped.")
        continue

    ## COMPUTING Ps, i.e., the simulated error parameters from the fake data
    ### Creating the noisy responses
    noisy_responses, th_mean_vec_Rs, th_cov_matrix_Rs = fd.fake_data(sequences, measurements, in_st, N, F)

    ## Theoretical mean vector and covariance matrix of Ps
    th_mean_vec_Ps = np.linalg.inv(L) @ (np.array(th_mean_vec_Rs) - np.array(list_R0))
    th_cov_matrix_Ps = np.linalg.inv(L) @ th_cov_matrix_Rs @ np.matrix.transpose(np.linalg.inv(L))

    mean_distance2 = np.matrix.trace(th_cov_matrix_Ps)
    # print("mean_distance2 = tr(L^(-1)*Sigma*L^(-T))= ", mean_distance2)

    list_theta.append(theta)
    list_cond_num.append(np.linalg.cond(L))
    list_mean_distance2.append(mean_distance2)

print("   ")
print("--- finished ---")

min_mean_distance2 = np.min(list_mean_distance2)
index = list_mean_distance2.index(min_mean_distance2)
theta_min = list_theta[index]

print("   ")
print("min_mean_distance2*N =", "{:.2f}".format(min_mean_distance2 * N))
print("<D^2>(2*pi-theta_min)*N =",
      "{:.2f}".format(opt.mean_distance2([2 * np.pi - theta_min for i in range(25)], N, F) * N))
print("<D^2>(pi/2)*N =", "{:.2f}".format(opt.mean_distance2([np.pi / 2 for i in range(25)], N, F) * N))
print("factor = ", "{:.2f}".format(min_mean_distance2 / opt.mean_distance2([np.pi / 2 for i in range(25)], N, F)))
print("index =", index)
print("theta_min / np.pi =", theta_min / np.pi)

list_theta = np.array(list_theta) / np.pi
theta_min = np.array(theta_min) / np.pi

# PLOTTING

f = plt.figure(figsize=(8, 4))
plt.subplot(1, 1, 1)
plt.plot(list_theta, list_mean_distance2, linestyle=' ', marker='.', color='g')
# plt.title("Mean squared distance as a function of theta for N=" + str(N))
plt.xlabel(r'$\theta/\pi$')
plt.ylabel(r'$\langle D^2\rangle$')
# plt.xlim([1.428 - 2, 1.428 + 2])
plt.grid(True, color='0.5', ls=':')
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
plt.ylim([0, 2 * 10 ** (-5)])
plt.tight_layout()
plt.show()

plt.subplot(2, 1, 1)
plt.plot(list_theta, list_cond_num, linestyle=' ', marker='.', color='g')
# plt.title("Condition number of L as a function of theta for N=" + str(N))
plt.xlabel(r'$\theta/\pi$')
plt.ylabel('Condition number of L')
plt.grid(True, color='0.5', ls=':')
# plt.xlim([0.1, 1.60])
plt.ylim([0, 20])

plt.subplot(2, 1, 2)
plt.plot(list_cond_num, list_mean_distance2, linestyle=' ', marker='.', color='g')
# plt.title("Mean squared distance as a function of the condition number of L for N=" + str(N))
plt.xlabel('Condition number of L')
plt.ylabel(r'$\langle D^2\rangle$')
plt.grid(True, color='0.5', ls=':')
plt.xlim([0, 0.5 * 10 ** 5])
plt.ylim([-1, 1 * 10 ** 1])

plt.tight_layout()
plt.show()