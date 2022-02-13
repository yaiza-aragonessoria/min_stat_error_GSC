# This file contains functions to simulate experiments that use GSC to calibrate a perturbed CNOT.

import numpy as np
from scipy.stats import norm
import sys
sys.path.insert(0, 'C:/Users/Yaiza/PycharmProjects/min_stat_error_GSC/functions/')
import basic_functions as bf

# This command suppresses printing of small floats
np.set_printoptions(suppress=True)

# Function that applies a concatenation of gates.
"""
gate_conc(gate_list,in_st)
    gate_list:=list of gates to apply on in_set. The list can have three kind of elements:
                - ["CNOT",con_qb], where CNOT specifies that the gate to be applied is the CNOT and con_qb=1,2 is the control qubit of the CNOT;
                - ["Rot",qb,theta], where Rot=Rx,Ry,Rz specifies the axis of the rotation, qb=1,2 is the qubit on which the rotation should be applied, and theta is the angle of the rotation;
                - ["U",qb,alpha,beta,gamma,delta], where U specifies that an arbitary 2-qb unitary should be applied on qb=1,2 with parameters (alpha,beta,gamma,delta);
                Ex: gate_list=[["CNOT",1],["U",1,0,0,0,0],["Rx",1,np.pi/2]]
    in_st:=list of one ore two complex arrays containing the initial state on which the gates in gate_list should be applied. It can have the two following forms:
            - in_st=[qb1,qb2], where qb1 and qb2 can be 2D column vectors (pure state) or 2x2 matrices (mixed state) and the global initial state is the tensor product of them.
            - in_st=[2qb_st], where 2qb_st is a 4D column vector (pure state) or a 4x4 matrix (mixed state) which already specifies the global initial state of the 2 qubits.
"""

def gate_conc(gate_list, in_st, state_check = "False"):
    # CHECK AND CONSTRUCTION OF THE STATE ~ Only if specified, the function checks if the input state is a valid state and builds the density matrix that will be used in the function.
    st = 0
    norm = 0
    type_st = 0
    pos_semidef = None
    if state_check == "True":  # The check of the state only takes place if the user specifies it.
        if len(in_st) == 1: # If only a global 2-qb state has been specified, the list state has only one element.
            st = in_st[0]
            if in_st[0].shape == (4, 1): # The specified state is a pure 2-qb state.
                valid=1
                type_st = "pure"
                norm = np.linalg.norm(st)
            elif in_st[0].shape == (4, 4): # The specified state is a mixed 2-qb state.
                valid=1
                type_st = "mixed"
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            else: # Any other shape of state[0] is not valid.
                valid=0
        elif len(in_st) == 2: # If the user specified a state for each qubit, the list in_st has two elements.
            if in_st[0].shape == (2, 1) and in_st[1].shape == (2, 1): # The specified states are two pure 1-qb states.
                valid=1
                type_st = "pure"
                st = np.kron(in_st[0], in_st[1])
                norm = np.linalg.norm(st)
            elif in_st[0].shape == (2, 1) and in_st[1].shape == (2, 2): # The specified states are a pure 1-qb state and a mixed 1-qb state, respectively.
                valid = 1
                type_st = "mixed"
                st1=np.kron(in_st[0],bf.dagga(in_st[0]))
                st2=in_st[1]
                st = np.kron(st1,st2)
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            elif in_st[0].shape == (2, 2) and in_st[1].shape == (2, 1): # The specified states are a mixed 1-qb state and a pure 1-qb state, respectively.
                valid = 1
                type_st = "mixed"
                st1 = in_st[0]
                st2 = np.kron(in_st[1], bf.dagga(in_st[1]))
                st = np.kron(st1, st2)
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            elif in_st[0].shape == (2, 2) and in_st[1].shape == (2, 2): # The specified states are two mixed 1-qb states.
                valid=1
                type_st = "mixed"
                st = np.kron(in_st[0], in_st[1])
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            else: # Any other shape of the elements of state is not valid.
                valid=0
        else: # If the list in_st has more than two elements, the syntax of in_st is not correct and the function sends an error message.
            valid = 2
    else:  # Unless specified the validity of the state is not check, so the function assumes it is a valid 2-qb density matrix.
        valid = 1
        norm = 1
        pos_semidef = True
        type_st = "mixed"
        st = in_st[0]
        # print("Not checking the validity of the state")

    # APPLICATION OF THE FUNCTION
    if valid==1: #The gates are only applied if the initial state is a valid 2-qb state.
        if 1.0-10**(-15) <= norm <= 1.0+10**(-15):  #The gates are only applied if the initial state has norm=1.
            i=0
            if type_st == "pure": #The function distinguishes between mixed and pure states because the formula of the resulting state is different.
                # Loop that applies the gates for an initial PURE state
                while i<len(gate_list):
                    G = bf.gate(gate_list[i])
                    st = G @ st
                    i+=1

                return st

            elif type_st == "mixed":
                # Loop that applies the gates for an initial MIXED state. It is analogous to the loop for pure states.
                if pos_semidef == True:
                    i=0
                    while i<len(gate_list):
                        G = bf.gate(gate_list[i])
                        st = G @ st @ bf.dagga(G)
                        i+=1

                else: # The function gives an error if the initial state is not positive semi-definite.
                    print("The initial state is not positive semi-definite.")
                    print("No state is returned.")

                    return None

                return st

            else: #The function gives an error if it could not assign the type of the state.
                print("Something went wrong. Please, check the initial state.")
                print("No state is returned.")
                return None

        else: #The function gives an error if the norm of the specified state is not 1.
            print("The initial state is not normalised.")
            print("No state is returned.")
            return None

    elif valid==0: #The function gives an error if the specified state has dimensions that don't match with a 2-qb state.
        print("The initial state is not a valid 2-qubit state.")
        print("No state is returned.")
        return None

    else: #The function gives an error if in_st has more than two elements.
        print("The syntax of the initial state is not correct.")
        print("No state is returned.")
        return None

# Measurement reponse of the measurement M on state, i.e., the function computes expected value of M.
def response(M,state,F,state_check = "False"):
    # CHECK AND CONSTRUCTION OF THE STATE ~ The function checks if the input state is a valid state and builds the density matrix that will be used in the function.
    norm = 0
    pos_semidef = None
    st = 0
    if state_check == "True": # The check of the state only takes place if the user specifies it.
        if len(state) == 1:  # If only a global 2-qb state has been specified, the list state has only one element.
            st = state[0]
            if state[0].shape == (4,1):  # The specified state is a pure 2-qb state.
                valid = 1
                norm = np.linalg.norm(st)
                pos_semidef = True
            elif state[0].shape == (4, 4): # The specified state is a mixed 2-qb state.
                valid = 1
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            else: # Any other shape of state[0] is not valid.
                valid = 0
        elif len(state) == 2:  # If the user specified a state for each qubit, the list in_st has two elements.
            if state[0].shape == (2, 1) and state[1].shape == (2,1):  # The specified states are two pure 1-qb states.
                valid = 1
                st = np.kron(state[0], state[1])
                norm = np.linalg.norm(st)
                pos_semidef = True
            elif state[0].shape == (2, 1) and state[1].shape == (2, 2): # The specified states are a pure 1-qb state and a mixed 1-qb state, respectively.
                valid = 1
                st1 = np.kron(state[0], bf.dagga(state[0]))
                st2 = state[1]
                st = np.kron(st1, st2)
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            elif state[0].shape == (2, 2) and state[1].shape == (2, 1): # The specified states are a mixed 1-qb state and a pure 1-qb state, respectively.
                valid = 1
                st1 = state[0]
                st2 = np.kron(state[1], bf.dagga(state[1]))
                st = np.kron(st1, st2)
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            elif state[0].shape == (2, 2) and state[1].shape == (2, 2): # The specified states are two mixed 1-qb states.
                valid = 1
                st = np.kron(state[0], state[1])
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            else: # Any other shape of the elements of state is not valid.
                valid = 0
        else:  # If the list state has more than two elements, the syntax of in_st is not correct and the function sends an error message.
            valid = 2
    else: # Unless specified the validity of the state is not check, so the function assumes it is a valid 2-qb density matrix.
        valid = 1
        norm = 1
        pos_semidef = True
        st = state[0]
        # print("Not checking the validity of the state")

    # APPLICATION OF THE FUNCTION
    if valid == 1:  # The function only runs if the initial state is a valid 2-qb state.
        if 1.0 - 10 ** (-15) <= norm <= 1.0 + 10 ** (-15):  # The function only runs if the initial state has norm=1.
            if pos_semidef == True: # The function only runs if the initial state is positive semi-definite.

                proj_p = 1 / 2 * (np.identity(4) + M) # Projector on the subspace of eigenvalue +1 of M.
                # proj_m = 1 / 2 * (np.identity(4) - M) # Projector on the subspace of eigenvalue -1 of M.

                p_p = np.real(np.matrix.trace(proj_p @ st)) # Probability of obtaining outcome +1 after measuring M on state.
                # p_m = np.real(np.matrix.trace(proj_m @ st)) # Probability of obtaining outcome +1 after measuring M on state.

                R = 2 *(F[0]*p_p+(1-F[1]) * (1 - p_p)) - 1 # Response (i.e., expected value) of the measurement M on state.

                # print("R =", R)

                return R, p_p

            else: # The function gives an error if the initial state is not positive semi-definite.
                print("The initial state is not positive semi-definite.")
                print("No state is returned.")

                return None

        else:  # The function gives an error if the norm of the specified state is not 1.
            print("The initial state is not normalised.")
            print("No state is returned.")
            return None

    elif valid == 0:  # The function gives an error if the specified state has dimensions that do not match with a 2-qb state.
        print("The initial state is not a valid 2-qubit state.")
        print("No state is returned.")
        return None

    else:  # The function gives an error if specified state has more than two elements.
        print("The syntax of the initial state is not correct.")
        print("No state is returned.")
        return None

# This function adds statistical errors to the responses according to the normal distribution with mean=0 and variance=N*(1/2*(R+1))*(1/2*(1-R)), where R is the response.
def noisy_responses(responses,N):
    i=0
    list_noisy_responses = []
    cov_matrix = np.zeros([len(responses),len(responses)])
    mean_vec = np.zeros(len(responses))
    while i < len(responses):
        variance = np.real(1-responses[i]**2) / (4*N) # Variance of the normal distribution associated to response[i].

        cov_matrix[i][i] = variance # Theoretical variance of the responses
        mean_vec[i] = responses[i] # Theoretical mean vector of the responses

        normal_distribution = norm(0, np.sqrt(variance)) # Normal distribution with mean=0 and variance=variance.

        error = normal_distribution.rvs() # Random sample of the normal distribution created above.

        noisy_R = responses[i] + error # Adding the statistical error (noise) to the (perturbed) responses.

        list_noisy_responses.append(noisy_R) # Adding the noisy response to the list of noisy responses.

        i += 1

    return list_noisy_responses, mean_vec, cov_matrix

# Applies sequences of gates to state, computes the response of measurements on the resulting state and gives the theoretical responses as well as noisy responses using the function noisy_responses()
def fake_data(sequences, measurements, state, N, F, statistical = True, state_check = False):
    # CHECK AND CONSTRUCTION OF THE STATE ~ The function checks if the input state is a valid state and builds the density matrix that will be used in the function.
    norm = 0
    pos_semidef = 0
    st = 0
    if state_check == True:  # The check of the state only takes place if the user specifies it.
        if len(state) == 1:  # If only a global 2-qb state has been specified, the list state has only one element.
            st = state[0]
            if state[0].shape == (4, 1):  # The specified state is a pure 2-qb state.
                valid = 1
                norm = np.linalg.norm(st)
                pos_semidef = True
            elif state[0].shape == (4, 4):  # The specified state is a mixed 2-qb state.
                valid = 1
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            else:  # Any other shape of state[0] is not valid.
                valid = 0
        elif len(state) == 2:  # If the user specified a state for each qubit, the list in_st has two elements.
            if state[0].shape == (2, 1) and state[1].shape == (2, 1):  # The specified states are two pure 1-qb states.
                valid = 1
                st = np.kron(state[0], state[1])
                norm = np.linalg.norm(st)
                pos_semidef = True
            elif state[0].shape == (2, 1) and state[1].shape == (2, 2):  # The specified states are a pure 1-qb state and a mixed 1-qb state, respectively.
                valid = 1
                st1 = np.kron(state[0], bf.dagga(state[0]))
                st2 = state[1]
                st = np.kron(st1, st2)
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            elif state[0].shape == (2, 2) and state[1].shape == (2, 1):  # The specified states are a mixed 1-qb state and a pure 1-qb state, respectively.
                valid = 1
                st1 = state[0]
                st2 = np.kron(state[1], bf.dagga(state[1]))
                st = np.kron(st1, st2)
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            elif state[0].shape == (2, 2) and state[1].shape == (2, 2):  # The specified states are two mixed 1-qb states.
                valid = 1
                st = np.kron(state[0], state[1])
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            else:  # Any other shape of the elements of state is not valid.
                valid = 0
        else:  # If the list state has more than two elements, the syntax of in_st is not correct and the function sends an error message.
            valid = 2
    else:  # Unless specified the validity of the state is not check, so the function assumes it is a valid 2-qb density matrix.
        valid = 1
        norm = 1
        pos_semidef = True
        st = state[0]
        #print("Not checking the validity of the state")

    # APPLICATION OF THE FUNCTION
    if valid == 1:  # The function only runs if the initial state is a valid 2-qb state.
        if 1.0 - 10 ** (-15) <= norm <= 1.0 + 10 ** (-15):  # The function only runs if the initial state has norm=1.
            if pos_semidef == True:  # The function only runs if the initial state is positive semi-definite.
                list_responses = []
                s = 0
                while s < len(sequences):
                    # print("s=", s + 1)
                    after_gates = gate_conc(sequences[s], [st])
                    # print("after_gates",s+1,"=", after_gates)

                    if np.all(after_gates == None): # The function gives an error if something went wrong with the function gate_conc() and no state was returned.
                        print("There is an error with the function gate_conc.")
                        break

                    R = response(measurements[s], [after_gates], F)[0] # Response of the measurement measurements[s] on the state after_gate.
                    # print("R[",s+1,"] =", bf.chop(R))

                    list_responses.append(R) # Adding the response to the list of responses.

                    s += 1

                # print("list_responses in fake data =", list_responses)
                if statistical == True:
                    list_noisy_responses, mean_vec, cov_matrix = noisy_responses(list_responses, N) # Adding statistical noise to all responses.
                else:
                    list_noisy_responses = list_responses  # No statistical noise is added.
                    cov_matrix = np.zeros([len(list_responses), len(list_responses)])
                    mean_vec = np.zeros(len(list_responses))

                return list_noisy_responses, mean_vec, cov_matrix

            else: # The function gives an error if the initial state is not positive semi-definite.
                print("The initial state is not positive semi-definite.")
                print("No state is returned.")

                return None

        else:  # The function gives an error if the norm of the specified state is not 1.
            print("The initial state is not normalised.")
            print("No state is returned.")
            return None

    elif valid == 0:  # The function gives an error if the specified state has dimensions that do not match with a 2-qb state.
        print("The initial state is not a valid 2-qubit state.")
        print("No state is returned.")
        return None

    else:  # The function gives an error if specified state has more than two elements.
        print("The syntax of the initial state is not correct.")
        print("No state is returned.")
        return None


#Error operator without approximation (Eq. 1 of GSC paper)
def EO(vec_p):
    i = 0
    error = np.identity(4)
    while i < len(vec_p):
        error = 1 / (np.sqrt(1 + vec_p[i] ** 2)) * (np.identity(4) - 1j * vec_p[i] * bf.paulis[i]) @ error
        i += 1
    return error

#Error operator (Eq. 1 of GSC paper) approximated to the first order.
def EOa(vec_p):
    i = 0
    error = 0
    while i < len(vec_p):
        error = error - 1j * vec_p[i] * bf.paulis[i]
        i += 1
    return np.identity(4)+error

# Constructing matrix L
def constructL(sequences, measurements, in_st, state_check="False"):
    # CHECK AND CONSTRUCTION OF THE STATE ~ Only if specified, the function checks if the input state is a valid state and builds the density matrix that will be used in the function.
    st = 0
    norm = 0
    pos_semidef = None
    if state_check == "True": # The check of the state only takes place if the user specifies it.
        if len(in_st) == 1:  # If only a global 2-qb state has been specified, the list state has only one element.
            st = in_st[0]
            if in_st[0].shape == (4, 1):  # The specified state is a pure 2-qb state.
                valid = 1
                norm = np.linalg.norm(st)
                pos_semidef = True
            elif in_st[0].shape == (4, 4): # The specified state is a mixed 2-qb state.
                valid = 1
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            else: # Any other shape of in_st[0] is not valid.
                valid = 0
        elif len(in_st) == 2:  # If the user specified a state for each qubit, the list in_st has two elements.
            if in_st[0].shape == (2, 1) and in_st[1].shape == (2,1):  # The specified states are two pure 1-qb states.
                valid = 1
                st = np.kron(in_st[0], in_st[1])
                norm = np.linalg.norm(st)
                pos_semidef = True
            elif in_st[0].shape == (2, 1) and in_st[1].shape == (2, 2): # The specified states are a pure 1-qb state and a mixed 1-qb state, respectively.
                valid = 1
                st1 = np.kron(in_st[0], bf.dagga(in_st[0]))
                st2 = in_st[1]
                st = np.kron(st1, st2)
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            elif in_st[0].shape == (2, 2) and in_st[1].shape == (2, 1): # The specified states are a mixed 1-qb state and a pure 1-qb state, respectively.
                valid = 1
                st1 = in_st[0]
                st2 = np.kron(in_st[1], bf.dagga(in_st[1]))
                st = np.kron(st1, st2)
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            elif in_st[0].shape == (2, 2) and in_st[1].shape == (2, 2): # The specified states are two mixed 1-qb states.
                valid = 1
                st = np.kron(in_st[0], in_st[1])
                norm = np.matrix.trace(st)
                pos_semidef = bf.is_pos_semidef(st)
            else: # Any other shape of the elements of state is not valid.
                valid = 0
        else:  # If the list in_st has more than two elements, the syntax of in_st is not correct and the function sends an error message.
            valid = 2
    else:  # Unless specified the validity of the state is not check, so the function assumes it is a valid 2-qb density matrix.
        norm = 1
        pos_semidef = True
        st = in_st[0]
        # print("Not checking the validity of the state")
        if st.shape == (4, 1):
            valid = 1
            st = np.kron(st, bf.dagga(st))
        elif st.shape == (4, 4):
            valid = 1
        else:
            valid = 0

    # APPLICATION OF THE FUNCTION
    if valid == 1:  # The function only runs if the initial state is a valid 2-qb state.
        if 1.0 - 10 ** (-15) <= norm <= 1.0 + 10 ** (-15):  # The function only runs if the initial state has norm=1.
            if pos_semidef == True: # The function only runs if the initial state is positive semi-definite.
                # INTRODUCING IDENTITY GATES TO sequences (to create "perfect" sequences)
                s=0
                while s < len(sequences):
                    # Locating CNOTS
                    loc = [0]
                    g = 0
                    while g < len(sequences[s]):
                        if sequences[s][g][0] == "ex":
                            loc.append(g)
                        g += 1
                    del loc[0]
                    loc.append(len(sequences[s]))

                    i = 0
                    while i < len(loc) - 1: # If a sequence has two consecutive CNOTs, an identity gate is inserted in between.
                        if loc[i] + 1 == loc[i + 1]:
                            sequences[s].insert(loc[i] + 1, ["Rx", 1, 0])
                        i += 1

                    if sequences[s][0][0] == "ex": # If a sequence starts a CNOT, an identity gate is inserted as the first.
                        sequences[s].insert(0, ["Rx", 1, 0])
                    if sequences[s][len(sequences[s]) - 1][0] == "ex": # If a sequence starts or finishes with a CNOT, an identity gate is inserted as the last gate.
                        sequences[s].append(["Rx", 1, 0])

                    s += 1

                    #CONSTRUCTING THE MATRIX L
                    L = np.zeros([15, 15])
                    s = 0
                    while s < len(sequences):
                        # Locating CNOTs
                        loc = [0]
                        g = 0
                        while g < len(sequences[s]):
                            if sequences[s][g][0] == "ex":
                                loc.append(g)
                            g += 1
                        del loc[0]
                        loc.append(len(sequences[s]))

                        c = len(loc) - 1  # Number of CNOTs

                        # Constructing the matrix Gs, which groups gates in between CNOTs
                        Gs = [np.identity(4) for _ in range(len(loc))]
                        g = 0
                        n = 0
                        while n < len(loc):
                            while g < loc[n]:
                                Gs[n] = bf.gate(sequences[s][g]) @ Gs[n]
                                g += 1
                            g += 1
                            n += 1

                        # Constructing the matrix Terms, which contains the ordered elements of each term that contributes to each matrix element of L (see notes of 26.02.2021 for a toy model)
                        u = 0
                        while u < 15:
                            # Initializing Terms
                            Terms = [[] for _ in range(2 * c)]

                            # Introducing Gs[i] to Terms
                            i = 0
                            while i < len(Terms):
                                j = 0
                                while j < len(Gs):
                                    Terms[i].append(Gs[j])
                                    if j != len(Gs) - 1:
                                        Terms[i].append(bf.CNOT(1))
                                    j += 1
                                i += 1

                            # Introducing dagga(Gs[i]) to Terms
                            i = 0
                            while i < len(Terms):
                                j = 0
                                while j < len(Gs):
                                    Terms[i].append(bf.dagga(Gs[j]))
                                    if j != len(Gs) - 1:
                                        Terms[i].append(bf.dagga(bf.CNOT(1)))
                                    j += 1
                                i += 1

                            # Introducing in_st to Terms
                            i = 0
                            while i < len(Terms):
                                Terms[i].insert(int(len(Terms[0]) / 2), in_st[0])
                                i += 1

                            # Introducing Paulis to Terms
                            i = 0
                            while i < len(Terms):
                                j = 0
                                while j < len(Terms[i]):
                                    if i < int(len(Terms) / 2):
                                        if j == 2 * i + 1:
                                            Terms[i].insert(2 * i + 1, bf.paulis[u])
                                    else:
                                        if j == 2 * i + 3:
                                            Terms[i].insert(2 * i + 3, bf.dagga(bf.paulis[u]))
                                    j += 1
                                i += 1

                            # Computing the matrix element L[s][u]
                            inside=None
                            i = 0
                            while i < len(Terms):
                                j = 0
                                if i < int(len(Terms) / 2):
                                    inside = Terms[i][len(Gs) + c + 1]
                                    while j < len(Terms[i]) / 2:
                                        if j == len(Terms[i]) / 2 - 1:
                                            inside = Terms[i][j] @ inside
                                        else:
                                            inside = Terms[i][j] @ inside @ Terms[i][j + len(Gs) + c + 2]
                                        j += 1
                                    L[s][u] = L[s][u] - (1j * np.matrix.trace(inside @ measurements[s])).real

                                elif i >= int(len(Terms) / 2):
                                    inside = Terms[i][len(Gs) + c]
                                    while j < len(Terms[i]) / 2:
                                        if j == len(Terms[i]) / 2 - 1:
                                            inside = inside @ Terms[i][j + len(Gs) + c + 1]
                                        else:
                                            inside = Terms[i][j] @ inside @ Terms[i][j + len(Gs) + c + 1]
                                        j += 1
                                    L[s][u] = L[s][u] + (1j * np.matrix.trace(inside @ measurements[s])).real
                                i += 1
                            u += 1
                        s += 1
                return L

            else: # The function gives an error if the initial state is not positive semi-definite.
                print("The initial state is not positive semi-definite.")
                print("No state is returned.")

                return None

        else:  # The function gives an error if the norm of the specified state is not 1.
            print("The initial state is not normalised.")
            print("No state is returned.")
            return None

    elif valid == 0:  # The function gives an error if the specified state has dimensions that don't match with a 2-qb state.
        print("The initial state is not a valid 2-qubit state.")
        print("No state is returned.")
        return None

    else:  # The function gives an error if specified state has more than two elements.
        print("The syntax of the initial state is not correct.")
        print("No state is returned.")
        return None
