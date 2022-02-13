# Optimisation of GSC over the 25 single-qubit rotations.

import numpy as np
import time
import nlopt as nlopt
import sys
sys.path.insert(0, 'C:/Users/Yaiza/PycharmProjects/min_stat_error_GSC/functions/')
import basic_functions as bf
import fake_data_functions as fd
import optimisation_functions as opt


np.set_printoptions(suppress=True)

# OPTIMISATION OVER 25 ANGLES
start_time = time.time()

M = 1*10**4 # Number of optimisations
F = [0.99, 0.98] # Measurement fidelities. If F=[1,1], the measurements are perfect.

list_result_code = [] # Initialising list for stopping criteria
list_opt_mean_distance2 = [] # Initialising list for mean squared distance
list_opt_vec_theta = [] # Initialising list for angles

for m in range(M):
    try: # When the optimiser runs into a RoundoffLimited error, the value of theta is skipped
        if m % 10 ** 2 == 0: # Counter to know at which point is the computation
            print("m =", m / (10 ** 2))

        vec_theta0 = np.random.random(25)*2*np.pi # The initial point of the optimiser is chosen randomly.

        inv = bf.is_invertible(np.array(opt.L(vec_theta0,F))) # Check if the initial value has a non-singular L.

        while inv == False:
            vec_theta0 = np.random.random(25) * 2 * np.pi
            inv = bf.is_invertible(opt.L(vec_theta0))

        # (LOCAL) OPTIMISATION
        optL = nlopt.opt(nlopt.LN_BOBYQA, 25) # Choice of the optimiser
        optL.set_maxtime(0.25*60*60) # Set max time for each optimisation
        optL.set_lower_bounds(np.zeros(25)) # Set 0 as lower bound for the angles
        optL.set_upper_bounds([2*np.pi for i in range(25)]) # Set 2*pi as upper bound for the angles
        optL.set_min_objective(opt.mysuperfunc) # Choose the function called opt.mysuperfunc1 to minimise
        optL.set_xtol_rel(1e-10) # Set tolerance for the angles
        opt_vec_theta = optL.optimize(vec_theta0) # Start optimisation with vec_theta0 as initial point and keep the optimal angles
        opt_mean_distance2 = optL.last_optimum_value() # Keep optimised mean squared distance
        result_code_L = optL.last_optimize_result() # Keep the stopping criteria

        list_opt_vec_theta.append(opt_vec_theta)
        list_opt_mean_distance2.append(opt_mean_distance2)
        list_result_code.append(result_code_L)

    except nlopt.RoundoffLimited: # Skipping values of theta that produce a RoundoffLimited error
        print("RoundoffLimited error at m=", m, ".")
        print("vec_theta0 =", vec_theta0)
        print("Value skipped.")

# SAVE DATA
## Save optimal angles, i.e., list_opt_vec_theta
file = open("data/opt/M=" + str(M) + "_F=" + str(F) + "_opt_theta.txt", "w")
np.savetxt(file, list_opt_vec_theta)
file.close()

## Save optimal mean squared distance, i.e., list_opt_mean_distance2
file = open("data/opt/M=" + str(M) + "_F=" + str(F) + "_opt_mean_distance2.txt", "w")
np.savetxt(file, list_opt_mean_distance2)
file.close()

## Save stopping reason of the optimisation, i.e., list_result_code
file = open("data/opt/M=" + str(M) + "_F=" + str(F) + "_opt_result_code.txt", "w")
np.savetxt(file, list_result_code)
file.close()

# print("list_opt_vec_theta = ", list_opt_vec_theta)
# print("list_opt_mean_distance2 = ", list_opt_mean_distance2)
# print("list_result_code =", list_result_code)

print("     ")
print("--- %s seconds ---" % (time.time() - start_time))