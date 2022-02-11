# This file contains some basic elements of quantum information.

import numpy as np
import scipy as sc
from scipy.stats import unitary_group

#Computational basis
up=np.array([[1],[0]],dtype=np.complex)
down=np.array([[0],[1]],dtype=np.complex)

# arbitrary (normalized) 2qb state on the computational basis
def state2(a,b):
    vec=a*up+b*down
    norm=np.linalg.norm(vec)
    return vec/norm

# arbitrary (normalized) 4qb state on the computational basis
def state4(a,b,c,d):
    vec=a*np.kron(up,up)+b*np.kron(up,down)+c*np.kron(down,up)+d*np.kron(down,down)
    norm=np.linalg.norm(vec)
    return vec/norm

# Pauli matrices
pauli_x=np.array([[0,1],[1,0]],dtype=complex)
pauli_y=np.array([[0,-1j],[1j,0]],dtype=complex)
pauli_z=np.array([[1,0],[0,-1]],dtype=complex)
id=np.identity(2)

# rotation matrices
def Rx(theta):
    exp=sc.linalg.expm(-1j*theta/2*pauli_x)
    return exp

def Ry(theta):
    exp=sc.linalg.expm(-1j*theta/2*pauli_y)
    return exp

def Rz(theta):
    exp=sc.linalg.expm(-1j*theta/2*pauli_z)
    return exp

# arbitrary 2-qb unitary
def U(alpha,beta,gamma,delta):
    U=np.exp(1j*alpha)*Rz(beta).dot(Ry(gamma).dot(Rz(delta)))
    return U

# CNOT
def CNOT(control_qubit):
    CNOT=None
    if control_qubit==1:
        CNOT=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    elif control_qubit==2:
        CNOT=np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
    return CNOT

# Exponential gate
def ex(H,a=1):
    U=None
    if a==1:
        U = sc.linalg.expm(1j * H)
    elif a==-1:
        U = sc.linalg.expm(-1j * H)
    else:
        print("The parameter a can only be +1 or -1.")
    return U

# dagga function
def dagga(matrix):
    return np.conjugate(matrix).transpose()

# Check if a matrix is positive semidefinite
def is_pos_semidef(x):
    if np.all(np.linalg.eigvals(x) >= 0 - 10 ** (-15)):
        pos_semidef=True
    else:
        pos_semidef=False
    return pos_semidef

# Check if a matrix is hermitian
def is_hermitian(x):
    H=np.all((x==dagga(x))==True)
    return H

# Check if a matrix is invertible
def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

# Combination of Pauli matrices acting on two qubits.
paulis = [np.kron(id,pauli_x),np.kron(id,pauli_y),np.kron(id,pauli_z),np.kron(pauli_x,id),np.kron(pauli_x,pauli_x),np.kron(pauli_x,pauli_y),np.kron(pauli_x,pauli_z),np.kron(pauli_y,id),np.kron(pauli_y,pauli_x),np.kron(pauli_y,pauli_y),np.kron(pauli_y,pauli_z),np.kron(pauli_z,id),np.kron(pauli_z,pauli_x),np.kron(pauli_z,pauli_y),np.kron(pauli_z,pauli_z)]

# Function that applies the gate specified via its parameters.
def gate(gate):
    # CNOT
    if gate[0] == "CNOT":
        if len(gate) == 2:
            if gate[1] == 1 or gate[1] == 2:
                con_qb = gate[1]
                G = CNOT(con_qb)
            else:
                print("Es tut mir leid, aber there are only two qubits in this world.")
                print("The qubit where the gate should be applied does not exist. Gates can only be applied on qubit 1 or 2.")
                G = None
        else:
            print("Incorrect syntax of gate.")
            G = None

    # Rotation around X axis
    elif gate[0] == "Rx":
        if len(gate) == 3:
            theta = gate[2]
            if gate[1] == 1:
                G = np.kron(Rx(theta), np.identity(2))
            elif gate[1] == 2:
                G = np.kron(np.identity(2), Rx(theta))
            else:
                print("Es tut mir leid, aber there are only two qubits in this world.")
                print("The qubit where the gate should be applied does not exist. Gates can only be applied on qubit 1 or 2.")
                G = None
        else:
            print("Incorrect syntax of gate.")
            G = None

    # Rotation around Y axis
    elif gate[0] == "Ry":
        if len(gate) == 3:
            theta = gate[2]
            if gate[1] == 1:  # The function checks on which qubit the rotation should be applied.
                G = np.kron(Ry(theta), np.identity(2))
            elif gate[1] == 2:
                G = np.kron(np.identity(2), Ry(theta))
            else:
                print("Es tut mir leid, aber there are only two qubits in this world.")
                print("The qubit where the gate should be applied does not exist. Gates can only be applied on qubit 1 or 2.")
                G = None
        else:
            print("Incorrect syntax of gate.")
            G = None

    # Rotation around Z axis
    elif gate[0] == "Rz":
        if len(gate) == 3:
            theta = gate[2]
            if gate[1] == 1:
                G = np.kron(Rz(theta), np.identity(2))
            elif gate[1] == 2:
                G = np.kron(np.identity(2), Rz(theta))
            else:
                print("Es tut mir leid, aber there are only two qubits in this world.")
                print("The qubit where the gate should be applied does not exist. Gates can only be applied on qubit 1 or 2.")
                G = None
        else:
            print("Incorrect syntax of gate.")
            G = None

    # Arbitrary unitary
    elif gate[0] == "U":
        if len(gate) == 6:
            alpha = gate[2]
            beta = gate[3]
            gamma = gate[4]
            delta = gate[5]
            if gate[1] == 1:
                G = np.kron(U(alpha, beta, gamma, delta), np.identity(2))
            elif gate[1] == 2:
                G = np.kron(np.identity(2), U(alpha, beta, gamma, delta))
            else:
                print("Es tut mir leid, aber there are only two qubits in this world.")
                print("The qubit where the gate should be applied does not exist. Gates can only be applied on qubit 1 or 2.")
                G = None

    # Exponential gate
    elif gate[0] == "ex":
        H = gate[1]
        if len(gate) == 2:
            G = ex(H)
        elif len(gate) == 3:
            if gate[2]==-1:
                G = ex(H,a=1)
            else:
                print("The parameter a can only be +1 or -1.")
        else:
            print("Incorrect syntax of gate.")
            print('The syntax to apply an exponential gate is ["ex",H] or ["ex",H,-1].')
            G = None

    # Random 2-qb gate
    elif gate[0] == "random2":
        if len(gate) == 1:
            G = unitary_group.rvs(4)
        else:
            print("Incorrect syntax of gate.")
            print('The syntax to apply a random 2-qb gate is ["random"].')

    # Random 1-qb gate
    elif gate[0] == "random1":
        if len(gate) == 1:
            # G = np.kron(unitary_group.rvs(2), unitary_group.rvs(2))
            b = np.random.normal(size=4)
            a = b / np.linalg.norm(b)
            U1 = a[0] * np.identity(2) + 1j * a[1] * pauli_x + 1j * a[2] * pauli_y + 1j * a[3] * pauli_z
            b = np.random.normal(size=4)
            a = b / np.linalg.norm(b)
            U2 = a[0] * np.identity(2) + 1j * a[1] * pauli_x + 1j * a[2] * pauli_y + 1j * a[3] * pauli_z
            G = np.kron(U1, U2)
        elif len(gate) == 2:
            if gate[1] == 1:
                # G = np.kron(unitary_group.rvs(2), np.identity(2))
                b = np.random.normal(size=4)
                a = b / np.linalg.norm(b)
                U1 = a[0] * np.identity(2) + 1j * a[1] * pauli_x + 1j * a[2] * pauli_y + 1j * a[3] * pauli_z
                G = np.kron(U1, np.identity(2))
            elif gate[1] == 2:
                # G = np.kron(np.identity(2), unitary_group.rvs(2))
                b = np.random.normal(size=4)
                a = b / np.linalg.norm(b)
                U2 = a[0] * np.identity(2) + 1j * a[1] * pauli_x + 1j * a[2] * pauli_y + 1j * a[3] * pauli_z
                G = np.kron(np.identity(2), U2)
            else:
                print('Es tut mir leid, aber there are only two qubits in this world.')
                print('The syntax to apply a random 1-qb unitary to each qubit is ["random1"].')
                print('If only one random 1-qb unitary should be applied, please specified in which qubit as ["random1", 1] or ["random1", 2].')
                G = None
        else:
            print("Incorrect syntax of gate.")
            print('The syntax to apply a random 1-qb unitary to each qubit is ["random1"].')
            print('If only one random 1-qb unitary should be applied, please specified in which qubit as ["random1", 1] or ["random1", 2].')

    else:
        print("Incorrect syntax of gate.")
        G = None

    return G