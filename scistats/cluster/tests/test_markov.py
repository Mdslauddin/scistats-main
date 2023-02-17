__all__ = ['hmmdecode', 'hmmestimate',  'hmmgenerate', 'hmmtrain', 'hmmiterbi']


import numpy as np


def hmm_posterior_states(X, transition_matrix, emission_matrix, initial_probabilities):
    n_states = transition_matrix.shape[0]
    n_samples = X.shape[0]

    # Initialize alpha matrix
    alpha = np.zeros((n_samples, n_states))
    alpha[0] = initial_probabilities * emission_matrix[:, X[0]]

    # Forward pass
    for t in range(1, n_samples):
        for j in range(n_states):
            alpha[t, j] = emission_matrix[j, X[t]] * np.sum(alpha[t-1] * transition_matrix[:, j])

    # Backward pass
    beta = np.zeros((n_samples, n_states))
    beta[-1] = 1
    for t in reversed(range(n_samples-1)):
        for i in range(n_states):
            beta[t, i] = np.sum(beta[t+1] * transition_matrix[i, :] * emission_matrix[:, X[t+1]])

    # Compute posterior state probabilities
    posterior = alpha * beta
    posterior /= np.sum(posterior, axis=1, keepdims=True)

    return posterior




def hmm_parameter_estimates(X, Z, n_states, n_features):
    # Estimate transition probabilities
    transition_counts = np.zeros((n_states, n_states))
    for i in range(1, len(Z)):
        transition_counts[Z[i-1], Z[i]] += 1
    transition_matrix = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)

    # Estimate initial probabilities
    initial_counts = np.zeros(n_states)
    initial_counts[Z[0]] += 1
    initial_probabilities = initial_counts / np.sum(initial_counts)

    # Estimate emission probabilities
    emission_counts = np.zeros((n_states, n_features))
    for i in range(len(Z)):
        emission_counts[Z[i], X[i]] += 1
    emission_matrix = emission_counts / np.sum(emission_counts, axis=1, keepdims=True)

    return transition_matrix, initial_probabilities, emission_matrix





def hmm_viterbi(X, transition_matrix, initial_probabilities, emission_matrix):
    # Initialize variables
    T = len(X)
    n_states, n_features = emission_matrix.shape
    delta = np.zeros((T, n_states))
    psi = np.zeros((T, n_states), dtype=int)
    states = np.zeros(T, dtype=int)

    # Initialize delta at time 0
    delta[0, :] = np.log(initial_probabilities) + np.log(emission_matrix[:, X[0]])

    # Iterate over time steps
    for t in range(1, T):
        for j in range(n_states):
            temp = delta[t-1, :] + np.log(transition_matrix[:, j]) + np.log(emission_matrix[j, X[t]])
            psi[t, j] = np.argmax(temp)
            delta[t, j] = temp[psi[t, j]]
    
    # Traceback to find the most likely sequence of states
    states[T-1] = np.argmax(delta[T-1, :])
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]

    # Map state indices to state labels
    state_labels = np.arange(n_states)
    state_map = dict(zip(state_labels, state_labels))
    emission_map = dict(zip(np.arange(n_features), X))

    # Map state and emission indices to labels
    states = np.vectorize(state_map.get)(states)
    emissions = np.vectorize(emission_map.get)(X)

    return states, emissions




def hmm_baum_welch(X, n_states, n_features, n_iterations=100, tol=1e-6):
    # Initialize HMM parameters randomly
    transition_matrix = np.random.rand(n_states, n_states)
    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)
    initial_probabilities = np.random.rand(n_states)
    initial_probabilities /= np.sum(initial_probabilities)
    emission_matrix = np.random.rand(n_states, n_features)
    emission_matrix /= np.sum(emission_matrix, axis=1, keepdims=True)

    # Run Baum-Welch algorithm
    for n in range(n_iterations):
        # E-step: Compute forward and backward probabilities
        alpha, beta = hmm_forward_backward(X, transition_matrix, initial_probabilities, emission_matrix)

        # M-step: Update HMM parameters
        transition_counts = np.zeros((n_states, n_states))
        for t in range(len(X)-1):
            transition_counts += alpha[t, :].reshape(-1, 1) * transition_matrix * beta[t+1, :] * emission_matrix[:, X[t+1]]
        transition_matrix = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)

        initial_probabilities = alpha[0, :] * beta[0, :]
        initial_probabilities /= np.sum(initial_probabilities)

        emission_counts = np.zeros((n_states, n_features))
        for t in range(len(X)):
            emission_counts[:, X[t]] += alpha[t, :] * beta[t, :]
        emission_matrix = emission_counts / np.sum(emission_counts, axis=1, keepdims=True)

        # Check for convergence
        if np.max(np.abs(transition_matrix - old_transition_matrix)) < tol and \
           np.max(np.abs(initial_probabilities - old_initial_probabilities)) < tol and \
           np.max(np.abs(emission_matrix - old_emission_matrix)) < tol:
            break

        # Save old HMM parameters
        old_transition_matrix = transition_matrix
        old_initial_probabilities = initial_probabilities
        old_emission_matrix = emission_matrix

    return transition_matrix, initial_probabilities, emission_matrix


import numpy as np

def hmm_viterbi(X, transition_matrix, initial_probabilities, emission_matrix):
    n_states, n_features = emission_matrix.shape

    # Initialize Viterbi variables
    T = len(X)
    delta = np.zeros((T, n_states))
    psi = np.zeros((T, n_states), dtype=int)

    # Compute initial delta values
    delta[0, :] = np.log(initial_probabilities) + np.log(emission_matrix[:, X[0]])

    # Run Viterbi algorithm
    for t in range(1, T):
        for j in range(n_states):
            # Compute state probabilities at time t
            state_probabilities = delta[t-1, :] + np.log(transition_matrix[:, j]) + np.log(emission_matrix[j, X[t]])

            # Find the most probable previous state
            psi[t, j] = np.argmax(state_probabilities)
            delta[t, j] = np.max(state_probabilities)

    # Backtrack to find most probable state path
    z = np.zeros(T, dtype=int)
    z[-1] = np.argmax(delta[-1, :])
    for t in range(T-2, -1, -1):
        z[t] = psi[t+1, z[t+1]]

    return z
