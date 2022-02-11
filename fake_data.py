# This project simulates an experiment that uses GSC to calibrate a perturbed CNOT

import numpy as np
import scipy as sc
import time

import functions.basic_functions as bf
import functions.fake_data_functions as fd
import functions.optimisation_functions as opt


## This command suppresses printing of small floats
np.set_printoptions(suppress=True)

## Setting printing options
np.set_printoptions(formatter={'float_kind':'{:.3e}'.format})

# SETTING THE ANGLES OF THE SINGLE-QUBIT ROTATIONS
F = [1, 1] # Measurement fidelities. If F=[1,1], the measurements are perfect


## Taking the optimised angles
## Reading optimised angles
M = 10 ** (2)  # Number of runs of the optimisation

with open("data/opt/M=" + str(M) + "_F=" + str(F) + "_opt_theta.txt") as f:
    list_vec_theta = [[num for num in line.split(' ')] for line in f]
    list_vec_theta = [[float(item) for item in line] for line in list_vec_theta]
f.close()

## Reading optimised mean sq distance
with open("data/opt/M=" + str(M) + "_F=" + str(F) + "_opt_mean_distance2.txt") as f:
    list_mean_distance2 = [num for num in f]
    list_mean_distance2 = [float(item) for item in list_mean_distance2]
f.close()

min = np.min(list_mean_distance2)
index = list_mean_distance2.index(min)
vec_theta = list_vec_theta[index]

while np.linalg.cond(opt.L(np.array(vec_theta), F)) > 50:
    del list_mean_distance2[index]
    del list_vec_theta[index]

    min = np.min(list_mean_distance2)
    index = list_mean_distance2.index(min)
    vec_theta = list_vec_theta[index]

# # Taking all angles equal to pi/2
# vec_theta = [ np.pi / 2 for i in range(25)]

# SETTING THE SEQUENCES, THE MEASUREMENTS AND THE INITIAL STATE OF EACH SETTING
s1 = [["CNOT",1],["Rx",1,vec_theta[14]]]
s2 = [["Rx",1,vec_theta[0]],["CNOT",1]]
s3 = [["CNOT",1],["Ry",1,vec_theta[15]]]
s4 = [["Ry",1,vec_theta[1]],["CNOT",1]]
s5 = [["CNOT",1],["Rx",2,vec_theta[16]]]
s6 = [["CNOT",1],["Ry",2,vec_theta[17]]]
s7 = [["Rx",1,vec_theta[2]],["CNOT",1],["Rx",1,vec_theta[18]]]
s8 = [["Rx",1,vec_theta[3]],["CNOT",1],["Ry",1,vec_theta[19]]]
s9 = [["Ry",1,vec_theta[4]],["CNOT",1],["Rx",1,vec_theta[20]]]
s10 = [["Rx",1,vec_theta[5]],["Rx",2,vec_theta[6]],["CNOT",1]]
s11 = [["Ry",1,vec_theta[7]],["CNOT",1],["Ry",1,vec_theta[21]]]
s12 = [["Ry",1,vec_theta[8]],["Ry",2,vec_theta[9]],["CNOT",1]]
s13 = [["Ry",2,vec_theta[10]],["CNOT",1],["Rx",2,vec_theta[22]]]
s14 = [["Rx",1,vec_theta[11]],["CNOT",1],["CNOT",1],["Ry",1,vec_theta[23]]]
s15 = [["Rx",1,vec_theta[12]],["Ry",2,vec_theta[13]],["CNOT",1],["Rx",2,vec_theta[24]]]

M30=np.kron(bf.pauli_z,np.identity(2))
M03=np.kron(np.identity(2),bf.pauli_z)

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

sequences = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15]
measurements = [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15]
in_st = [np.kron(np.kron(bf.up, bf.up), bf.dagga(np.kron(bf.up, bf.up)))]


# PARAMETERS OF THE SIMULATION
N = 10**(6) # Number of runs of the "experiment in the lab"
J = 10**4 # Number of simulations to compute the histogram, i.e., to do an statistical analysis


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

## Computing the theoretical responses, i.e., R0, which have systematic errors, but no statistical error
s = 0
list_R0 = []
while s < len(sequences):
    # print("s=", s + 1)
    after_gates = fd.gate_conc(sequences[s], in_st) # Computing the state after applying the gates of sequence s
    # print("after_gates",s+1,"=", after_gates)

    R0 = fd.response(measurements[s], [after_gates], F)[0]  # Response of the measurement measurements[s] on the state after_gate.
    # print("R0[",s+1,"] =", R0)

    list_R0.append(R0)  # Adding the response to the list of responses.

    s += 1

print("vec_theta =", vec_theta)
print("list_R0 =", str(['%.4f' % elem for elem in list_R0]).replace("'", ""))

## As the CNOT is the gate we want to calibrate, we change the sequences  such that the code applies a perturbed CNOT, i.e., a CÑOT, and not a CNOT.
s = 0
while s < len(sequences):
    g = 0
    while g < len(sequences[s]):
        if sequences[s][g][0] == "CNOT":
            sequences[s][g] = ["ex", HCÑOT]
        g += 1
    s += 1

# print("list_thR =", str(['%.4f' % elem for elem in list_thR]).replace("'", ""))

# CONSRTUCTING MATRIX L
## Constructing matrix L (see GSC paper)
L = (F[0]+F[1]-1)*fd.constructL(sequences, measurements, in_st)
# print("L=", L)
print("cond(L) =", np.linalg.cond(L))

## Checking if L is singular
if bf.is_invertible(L) == False:
    print("L is not invertible.")
    print("Process stopped.")
    exit()

start_time = time.time()

Rs = np.zeros([15, J]) # Initialise the list of responses
Ps = np.zeros([15, J]) # Initialise the list of error parameters

j = 0
while j < J:
    if j % 10 ** 3 == 0:
        print("j =", j)

    ## COMPUTING Ps, i.e., the simulated error parameters from the fake data
    ### Creating the noisy responses
    noisy_responses, th_mean_vec_Rs, th_cov_matrix_Rs = fd.fake_data(sequences, measurements, in_st, N, F)
    np.matrix.transpose(Rs)[j] = noisy_responses

    ### Solving the inverse problem R=R0+L·Ps => Ps=L^(-1)(R-R0), where L is the L matrix, noisy_responses a vector with the simulated noise responses and Ps a vector with the error parameters.
    np.matrix.transpose(Ps)[j] = np.linalg.solve(L, np.array(noisy_responses) - np.array(list_R0))

    j += 1

## Experimental mean vector and covariance matrix of Ps
exp_mean_vec_Ps = np.mean(Ps, axis=1)
exp_cov_matrix_Ps = np.cov(Ps)

## Theoretical mean vector and covariance matrix of Ps
th_mean_vec_Ps = np.linalg.inv(L) @ (np.array(th_mean_vec_Rs) - np.array(list_R0))
th_cov_matrix_Ps = np.linalg.inv(L) @ th_cov_matrix_Rs @ np.matrix.transpose(np.linalg.inv(L))

## Experimental mean vector and covariance matrix of Ps
exp_mean_vec_Rs = np.mean(Rs, axis=1)
exp_cov_matrix_Rs = np.cov(Rs)

## SAVE DATA
# """
# Save ps
file = open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_pss.txt", "w")
file.write("# ps for N=" + str(N) + " and J=" + str(J) + "\n")
np.savetxt(file, ps)
file.close()

# Save mean vector of noisy responses
file_mean_vector = open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_mean_vector.txt", "w")
file_mean_vector.write("# Theoretical mean vector of the noisy responses for N=" + str(N) + " and J=" + str(J) + "\n")
np.savetxt(file_mean_vector, th_mean_vec_Rs)

# Save covariance matrix of noisy responses
file_cov_matrix = open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_cov_matrix.txt", "w")
file_cov_matrix.write("# Theoretical covariance matrix of the noisy responses for N=" + str(N) + " and J=" + str(J) + "\n")
np.savetxt(file_cov_matrix, th_cov_matrix_Rs)

# Save mean vector for Ps
file_mean_vector.write("# Theoretical mean vector of Ps for N=" + str(N) + " and J=" + str(J) + "_F=" + str(F) + "\n")
np.savetxt(file_mean_vector, th_mean_vec_Ps)
file.close()

# Save covariance matrix for Ps
file_cov_matrix.write("# Theoretical covariance matrix of Ps for N=" + str(N) + " and J=" + str(J) + "_F=" + str(F) + "\n")
np.savetxt(file_cov_matrix, th_cov_matrix_Ps)

# Save Rs
file = open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_Rs.txt", "w")
file.write("# Rs for N=" + str(N) + " and J=" + str(J) + "_F=" + str(F) + "\n")
np.savetxt(file, Rs)
file.close()

# Save Ps
file = open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_Ps.txt", "w")
file.write("# Ps for N=" + str(N) + " and J=" + str(J) + "\n")
np.savetxt(file, Ps)
file.close()

# Save vec_theta
file = open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_vec_theta.txt", "w")
file.write("# vec_theta for N=" + str(N) + " and J=" + str(J) + "\n")
np.savetxt(file, vec_theta)
file.close()

# Save c
file = open("data/opt_lab/N=" + str(N) + "_J=" + str(J) + "_F=" + str(F) + "_vec_theta.txt", "w")
file.write("# vec_theta for N=" + str(N) + " and J=" + str(J) + "\n")
np.savetxt(file, c)
file.close()
# """

## PRINT DATA
print("     ")
print("--- %s seconds ---" % (time.time() - start_time))
print("   ")
print("exp_mean_vec_Rs")
print(exp_mean_vec_Rs)
print("th_mean_vec_Rs")
print(th_mean_vec_Rs)
print("  ")
print("exp_cov_matrix_Rs =", exp_cov_matrix_Rs)
print("th_cov_matrix_Rs =", th_cov_matrix_Rs)
print("   ")
print("exp_mean_vec_Ps")
print(exp_mean_vec_Ps)
print("th_mean_vec_Ps")
print(th_mean_vec_Ps)
print("real error parameters, i.e., ps")
print(ps)
print("  ")
print("exp_cov_matrix_Ps =", exp_cov_matrix_Ps)
print("th_cov_matrix_Ps =", th_cov_matrix_Ps)

