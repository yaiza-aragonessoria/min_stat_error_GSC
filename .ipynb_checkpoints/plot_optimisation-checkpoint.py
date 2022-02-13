# This file plots from a file

import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'C:/Users/Yaiza/PycharmProjects/min_stat_error_GSC/functions/')
import basic_functions as bf
import fake_data_functions as fd
import optimisation_functions as opt

# SETTING PARAMETERS OF PLOTTING
## Setting print parameters
np.set_printoptions(formatter={'float_kind':'{:.4e}'.format}) # Limiting the number of decimals shown
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)}) # Limiting the number of decimals shown
np.set_printoptions(formatter={'float64': lambda x: "{0:0.4f}".format(x)}) # Limiting the number of decimals shown

## Setting plot parameters
plt.rcParams.update({'font.size': 16}) # Font size
plt.rcParams['mathtext.fontset'] = 'stix' # Font similar to latex font
plt.rcParams['font.family'] = 'STIXGeneral' # Font similar to latex font

## Setting the path where the plots will be saved
my_path = os.path.abspath('C:/Users/Yaiza/PycharmProjects/min_stat_error/plot/') # Figures out the absolute path for you in case your working directory moves around.


# READING DATA
## Data with F=[1,1] and angles from 0 to 2pi
M = 1*10**2
F=[1, 1]

### Reading optimised angles
with open("data/opt/M=" + str(M) + "_F=" + str(F) + "_opt_theta.txt") as f:
    list_theta_F1 = [[num for num in line.split(' ')] for line in f]
    list_theta_F1 = [[float(item) for item in line] for line in list_theta_F1]
f.close()

### Reading optimised mean sq distance
with open("data/opt/M=" + str(M) + "_F=" + str(F) + "_opt_mean_distance2.txt") as f:
    list_mean_distance2_F1 = [num for num in f]
    list_mean_distance2_F1 = [float(item) for item in list_mean_distance2_F1]
f.close()

## Data with F!=1 from 0 to 2pi
M = 1*10**2
F = [0.99, 0.98]

### Reading optimised angles F!=1 from 0 to 2pi
with open("data/opt/M=" + str(M) + "_F=" + str(F) + "_opt_theta.txt") as f:
    list_theta_F = [[num for num in line.split(' ')] for line in f]
    list_theta_F = [[float(item) for item in line] for line in list_theta_F]
f.close()

### Reading optimised mean sq distance F!=1 with angles from 0 to 2pi
with open("data/opt/M=" + str(M) + "_F=" + str(F) + "_opt_mean_distance2.txt") as f:
    list_mean_distance2_F = [num for num in f]
    list_mean_distance2_F = [float(item) for item in list_mean_distance2_F]
f.close()

## Creating a list with numbers to be the x-axis of the plots. Each number denotes a
label = []
for n in range(25):
    label.append(n + 1)

print("      ")
print("#------------- PERFECT MEASUREMENTS -------------#")
print("      ")

F = [1,1]

print("      ")
print("PLOT OPTIMAL ANGLES with F=[1,1] from 0 to 2*pi")
print("      ")

## Only the angles which give a mean squared distance smaller than 8.4231e-07 will be plot for clarity
plot_list_mean_distance2_F1 = [x for x in list_mean_distance2_F1 if x <= 8.4231e-07]

print("len(plot_list_mean_distance2_F1) = ", len(plot_list_mean_distance2_F1))

## We want to remove the angles that give an unstable L^(-1) matrix
## For that we first create a list which keeps the position in the global list, i.e., in list_mean_distance2_F1, of the values to plot, i.e., in plot_list_mean_distance2_F1
indexes_F1 = []
for i in range(len(plot_list_mean_distance2_F1)):
    index_F1 = list_mean_distance2_F1.index(plot_list_mean_distance2_F1[i])
    indexes_F1.append(index_F1)

## Then, we create a list of the angles to plot
plot_list_theta_F1 = []
for i in indexes_F1:
    plot_list_theta_F1.append(list_theta_F1[i])

## We look for the minimal mean squared distance and correspoinding angle and index
min_F1 = np.min(plot_list_mean_distance2_F1)
index_F1 = plot_list_mean_distance2_F1.index(min_F1)
theta_min_F1 = plot_list_theta_F1[index_F1]

## If the minimal mean squared distance requires an unstable L^(-1), we discard it and take the next minimal value.
while np.linalg.cond(opt.L(np.array(theta_min_F1),F))>50:
    del plot_list_mean_distance2_F1[index_F1]
    del plot_list_theta_F1[index_F1]
    del indexes_F1[index_F1]

    min_F1 = np.min(plot_list_mean_distance2_F1)
    index_F1 = plot_list_mean_distance2_F1.index(min_F1)
    theta_min_F1 = plot_list_theta_F1[index_F1]

## For clarity, we plot the angles divided by pi
plot_list_theta_F1 = np.array(plot_list_theta_F1) / np.pi

print("\n")
print("min =", min_F1)
print("mean =", opt.mean_distance2(theta_min_F1,10**6,F))
print("index_F1 =", index_F1)
print("theta_min_F1 / pi =", np.array(theta_min_F1) / np.pi)
print("cond =", np.linalg.cond(opt.L(np.array(theta_min_F1), [1, 1])))
print("\n")
print("after finding a valid minumum...")
print("len(plot_list_mean_distance2_F1) =", len(plot_list_mean_distance2_F1))
print("len(plot_list_theta_F1) =", len(plot_list_theta_F1))
print("len(indexes_F1) =", len(indexes_F1))

## Plotting optimal angles
plt.subplot(1, 1, 1)
plt.plot(label, np.array(theta_min_F1) / np.pi, linestyle=' ', marker='o', alpha=1, color='green')
plt.title("Optimal angles with F=" + str(F))
plt.xlabel('i')
plt.ylabel('$\\theta_i/\\pi$')
plt.grid(True, color='0.5', ls=':')
plt.xticks(range(1, len(theta_min_F1)+1))
plt.tight_layout()
plt.ylim([0, 2])
plt.show()

print("      ")
print("PLOT FOLDED ANGLES with F=[1,1] from 0 to 2*pi")
print("      ")

# Creating the folded angles, i.e., applying the map theta_i --> 2*pi-theta_i for all theta_i > pi
vec_theta_F1 = np.array(theta_min_F1) / np.pi
fol_theta_F1 = np.array([2 - theta if theta > 1 else theta for theta in vec_theta_F1])

print("vec_theta_F1 =", np.array(vec_theta_F1))
print("mean =", opt.mean_distance2(vec_theta_F1*np.pi,10**6,F))
print("fol_theta_F1 =", np.array(fol_theta_F1))
print("\n")

print("mean_distance2(vec_theta_F1) =", opt.mean_distance2(vec_theta_F1*np.pi, 10 ** 6, F))
print("mean_distance2(fol_theta_F1) =", opt.mean_distance2(fol_theta_F1*np.pi, 10 ** 6, F))
print("\n")

## Plotting folded angles
plt.subplot(1, 1, 1)
plt.plot(label, fol_theta_F1, linestyle=' ', marker='o', alpha=1, color='green')
plt.title("Folded angles with F=" + str(F))
plt.xlabel('i')
plt.ylabel('$\\theta_i/\\pi$')
plt.grid(True, color='0.5', ls=':')
plt.xticks(range(1, len(fol_theta_F1)+1))
plt.tight_layout()
plt.ylim([0, 2])
plt.show()

print("      ")
print("PLOT MODIFIED ANGLES with F=[1,1] from 0 to 2*pi")
print("      ")

## Initialising the list of modified angles
mod_theta_F1 = [0 for i in range(25)]

## List of angles that do not modify the mean squared distance
deg_angles = [1, 2, 6, 9, 15, 16, 17, 18]

## Creating the modified angles, i.e., assigning 0.5*pi to all angles in deg_angles and the rest keep the value of the folded angles
for i in range(25):
    if i + 1 in deg_angles:
        mod_theta_F1[i] = 0.5
    else:
        mod_theta_F1[i] = fol_theta_F1[i]

mod_theta_F1 = np.array(mod_theta_F1)

print("fol_theta_F1 =", np.array(fol_theta_F1))
print("mod_theta_F1 =", np.array(mod_theta_F1))
print("\n")

print("mean_distance2(fol_theta_F1) =", opt.mean_distance2(fol_theta_F1 * np.pi, 10 ** 6, F))
print("mean_distance2(mod_theta_F1) =", opt.mean_distance2(mod_theta_F1 * np.pi, 10 ** 6, F))
print("difference=", np.abs(opt.mean_distance2(fol_theta_F1 * np.pi, 10 ** 6, F) - opt.mean_distance2(mod_theta_F1 * np.pi, 10 ** 6, F)))
print("\n")

## Plotting the modified angles
plt.subplot(1, 1, 1)
# plt.plot(label, fol_theta_F1, linestyle=' ', marker='o', alpha=0.5, color='orange')
plt.plot(label, mod_theta_F1, linestyle=' ', marker='o', alpha=1, color='green')
plt.title("Modified angles with F=" + str(F))
plt.xlabel('i')
# plt.ylabel('mod_theta5_F1')
plt.ylabel('$\\theta_i/\\pi$')
plt.grid(True, color='0.5', ls=':')
plt.xticks(range(1, len(plot_list_theta_F1[0]) + 1))
plt.tight_layout()
plt.ylim([0, 2])
plt.show()


print("      ")
print("PLOT OF THE DEGENERACY of the optimal mean squared distance for F=[1,1]")
print("      ")
## The angles theta1, theta2, theta6, theta9, theta15, theta16, theta17, theta18 can be chosen randomly, as we see below.

## Starting the plot and the axis
fig = plt.figure()
ax = fig.add_subplot(111)

## Computing mean squared distances for random values of theta1, theta2, theta6, theta9, theta15, theta16, theta17, theta18.
Y = 10 ** 5 # Number of random angles that will be considered
list_mean_distance2 = []
for i in range(Y):
    if i % 10 ** 4 == 0: # Counter to know who the computation is going.
        print("i =", i)
        print("\n")

    ## Initialising the list of modified angles
    mod_theta_F1 = [0 for i in range(25)]

    ## List of angles that do not modify the mean squared distance
    deg_angles = [1, 2, 6, 9, 15, 16, 17, 18]

    ## For al angles in deg_angles a random value is assigned. For the rest, the value in vec_theta_F1 is kept.
    for i in range(25):
        if i + 1 in deg_angles:
            mod_theta_F1[i] = 2*np.random.random()
        else:
            mod_theta_F1[i] = vec_theta_F1[i]

    mod_theta_F1 = np.array(mod_theta_F1)

    ## We keep the mean squared distance that has a stable matrix L^(-1) to plot it later.
    if np.linalg.cond(opt.L(np.array(mod_theta_F1),F))>10**12:
        print("cond(L) =", np.linalg.cond(opt.L(np.array(mod_theta_F1), F))) # If L^(-1) is unstable, we plot its condition number.
    else:
        ## When the difference mean squared distance with random variables and the minimal mean squared distance is bigger than 10**(-12), the values are printed.
        if np.abs(opt.mean_distance2(mod_theta_F1 * np.pi, 10 ** 6, [1, 1]) - min_F1) > 10 ** (-12):
            print("cond(L) =", np.linalg.cond(opt.L(np.array(mod_theta_F1), F)))
            print("mod = ", mod_theta_F1)
            print("<D^2>_min =", min_F1)
            print("<D^2>_mod =", opt.mean_distance2(mod_theta_F1 * np.pi, 10 ** 6, [1, 1]))
            print("diff =", np.abs(opt.mean_distance2(mod_theta_F1 * np.pi, 10 ** 6, [1, 1]) - min_F1))
            print("\n")

        D2 = opt.mean_distance2(mod_theta_F1 * np.pi, 10 ** 6, [1, 1]) # Computing the mean squared distance
        list_mean_distance2.append(D2) # Keeping the mean squared distance
        plt.plot(label, mod_theta_F1, linestyle=' ', marker='o', alpha=0.5, color='green') # Plotting the angles corresponding tot he mean squared distance

## Setting parameters of the plot of the angles
ax.set_xlabel('i', fontsize=24)
ax.set_ylabel('$\\theta_i/\\pi$', fontsize=24)

### I want max x axis to be 26
ax.set_xlim(0, 26)
### I want max y axis to be 2
ax.set_ylim(0, 2)

### Set major ticks for x axis
major_xticks = range(1, len(plot_list_theta_F1[0])+1,2)

### Set major ticks for y axis
major_yticks = np.arange(0, 2.5, 0.5)

### I want minor ticks for x axis
minor_xticks = np.arange(2, 26, 2)

### I want minor ticks for y axis
minor_yticks = np.arange(0, 2.5, 0.5)

### Specify tick label size
ax.tick_params(axis = 'both', which = 'major', labelsize=24)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 0) # Suppress minor tick labels

### Specify which ticks should be shown for the x-axis
ax.set_xticks(major_xticks)
ax.set_xticks(minor_xticks, minor = True)

### Specify which ticks should be shown for the y-axis
ax.set_yticks(major_yticks)
ax.set_yticks(minor_yticks, minor = True)

### Set both ticks to be outside
ax.tick_params(which = 'both', direction = 'out')

## Specify different settings for major and minor grids
ax.grid(which = 'minor', alpha = 0.5)
ax.grid(which = 'major', alpha = 0.5)

plt.tight_layout()
plt.title("Degeneracy of <D> for F=" + str(F))
plt.show()

# Plotting the mean squared distances
plt.plot(list_mean_distance2, linestyle=' ', marker='o', alpha=0.5, color='green')
plt.title("<D> with random $\\theta_1$, $\\theta_2$, $\\theta_6$, $\\theta_9$, $\\theta_{15}$, $\\theta_{16}$, $\\theta_{17}$, $\\theta_{18}$\n")
plt.tight_layout()
plt.show()


print("      ")
print("#------------- IMPERFECT MEASUREMENTS -------------#")
print("      ")
## Code analogous to the perfect-measurement one (without degeneracy as it is broken for imperfect measurements)

F = [0.99, 0.98]

print("      ")
print("# PLOT OPTIMAL ANGLES with F=[0.99, 0.98] from 0 to 2*pi")
print("      ")

plot_list_mean_distance2_F = [x for x in list_mean_distance2_F if x <= 8.9456e-07]

print("len(plot_list_mean_distance2_F) = ", len(plot_list_mean_distance2_F))

indexes_F = []
for i in range(len(plot_list_mean_distance2_F)):
    index_F = list_mean_distance2_F.index(plot_list_mean_distance2_F[i])
    indexes_F.append(index_F)

plot_list_theta_F = []
for i in indexes_F:
    plot_list_theta_F.append(list_theta_F[i])

min_F = np.min(plot_list_mean_distance2_F)
index_F = plot_list_mean_distance2_F.index(min_F)
theta_min_F = plot_list_theta_F[index_F]

while np.linalg.cond(opt.L(np.array(theta_min_F), F)) > 50:
    del plot_list_mean_distance2_F[index_F]
    del plot_list_theta_F[index_F]
    del indexes_F[index_F]

    min_F = np.min(plot_list_mean_distance2_F)
    index_F = plot_list_mean_distance2_F.index(min_F)
    theta_min_F = plot_list_theta_F[index_F]

plot_list_theta_F = np.array(plot_list_theta_F) / np.pi

print("\n")
print("min =", min_F)
print("mean =", opt.mean_distance2(theta_min_F, 10 ** 6, F))
print("index_F =", index_F)
print("theta_min_F / pi =", np.array(theta_min_F) / np.pi)
print("cond =", np.linalg.cond(opt.L(np.array(theta_min_F), [1, 1])))
print("\n")
print("after finding a valid minumum...")
print("len(plot_list_mean_distance2_F) =", len(plot_list_mean_distance2_F))
print("len(plot_list_theta_F) =", len(plot_list_theta_F))
print("len(indexes_F) =", len(indexes_F))

## Plotting optimal angles
plt.subplot(1, 1, 1)
plt.plot(label, np.array(theta_min_F) / np.pi, linestyle=' ', marker='o', alpha=1, color='green')
plt.title("Optimal angles with F=" + str(F))
plt.xlabel('i')
plt.ylabel('$\\theta_i/\\pi$')
plt.grid(True, color='0.5', ls=':')
plt.xticks(range(1, len(theta_min_F) + 1))
plt.tight_layout()
plt.ylim([0, 2])
plt.show()


print("      ")
print("# PLOT FOLDED ANGLES with F=[0.99,0.98] from 0 to 2*pi")
print("      ")

vec_theta_F = np.array(theta_min_F) / np.pi
fol_theta_F = np.array([2 - theta if theta > 1 else theta for theta in vec_theta_F])

print("vec_theta_F =", np.array(vec_theta_F))
print("mean =", opt.mean_distance2(vec_theta_F*np.pi,10**6, F))
print("fol_theta_F =", np.array(fol_theta_F))
print("\n")

print("mean_distance2(vec_theta_F, [0.99,0.98]) =", opt.mean_distance2(vec_theta_F*np.pi, 10 ** 6, F))
print("mean_distance2(fol_theta_F, [0.99,0.98]) =", opt.mean_distance2(fol_theta_F*np.pi, 10 ** 6, F))
print("\n")
print("mean_distance2(vec_theta_F, [1,1]) =", opt.mean_distance2(vec_theta_F*np.pi, 10 ** 6, [1,1]))
print("mean_distance2(fol_theta_F, [1,1]) =", opt.mean_distance2(fol_theta_F*np.pi, 10 ** 6, [1,1]))
print("\n")

## Plotting folded optimal angles
plt.subplot(1, 1, 1)
plt.plot(label, fol_theta_F, linestyle=' ', marker='o', alpha=1, color='green')
plt.title("Folded angles with F=" + str(F))
plt.xlabel('i')
plt.ylabel('$\\theta_i/\\pi$')
plt.grid(True, color='0.5', ls=':')
plt.xticks(range(1, len(fol_theta_F)+1))
plt.tight_layout()
plt.ylim([0, 2])
plt.show()


print("      ")
print("# PLOT MODIFIED ANGLES with F=[0.99,0.98] from 0 to 2*pi")
print("      ")

mod_theta_F = [0 for i in range(25)]

deg_angles = [1, 2, 6, 9, 15, 16, 17, 18]

for i in range(25):
    if i + 1 in deg_angles:
        mod_theta_F[i] = 0.5
    else:
        mod_theta_F[i] = fol_theta_F[i]

mod_theta_F = np.array(mod_theta_F)

print("fol_theta_F =", np.array(fol_theta_F))
print("mod_theta_F =", np.array(mod_theta_F))
print("\n")

print("mean_distance2(vec_theta_F)*10**6 =", "{:.2f}".format(opt.mean_distance2(vec_theta_F * np.pi, 10 ** 6, F) * 10 ** 6))
print("mean_distance2(fol_theta_F)*10**6 =", "{:.2f}".format(opt.mean_distance2(fol_theta_F * np.pi, 10 ** 6, F) * 10 ** 6))
print("mean_distance2(mod_theta_F)*10**6 =", "{:.2f}".format(opt.mean_distance2(mod_theta_F * np.pi, 10 ** 6, F) * 10 ** 6))
print("difference=", np.abs(opt.mean_distance2(fol_theta_F * np.pi, 10 ** 6, F) - opt.mean_distance2(mod_theta_F * np.pi, 10 ** 6, F)))
print("\n")

# Plotting folded optimal angles
plt.subplot(1, 1, 1)
# plt.plot(label, fol_theta_F, linestyle=' ', marker='o', alpha=0.5, color='orange')
plt.plot(label, mod_theta_F, linestyle=' ', marker='o', alpha=1, color='green')
plt.title("Modified angles with F=" + str(F))
plt.xlabel('i')
plt.ylabel('$\\theta_i/\\pi$')
# plt.ylabel('mod_theta_F')
plt.grid(True, color='0.5', ls=':')
plt.xticks(range(1, len(mod_theta_F) + 1))
plt.tight_layout()
plt.ylim([0, 2])
plt.show()