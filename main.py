#Filename:	main.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 17 Nov 2020 07:43:41  WIB

import numpy as np
from categorical_scatter import categorical_scatter_2d
from load_data import load_mnist

NUM_POINTS = 200
CLASSES_TO_USE = [0, 1, 8]
PERPLEXITY = 20
SEED = 1
MOMENTUM = 0.9
LEARNING_RATE = 10.
NUM_ITERS = 500
TSNE = True
NUM_PLOTS = 5

def neg_square_euc_distance(X):
    """
    Compute matrix containing negative squared euclidean distance for all pairs
    of points in the input matrix X

    # Arguments:
    X: matrix of size N * D
    # Return
    N*N matrix D, with entry D_ij = negative squared
    euclidean distance between rows X_i, and X_j

    """
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return -D

def softmax(X, diag_zero = True):

    e_x = np.exp(X - np.max(X, axis = 1).reshape([-1, 1]))

    if diag_zero:
        np.fill_diagonal(e_x, 0)

    e_x = e_x + 1e-8

    return e_x / e_x.sum(axis = 1).reshape([-1, 1])

def calc_prob_matrix(distances, sigmas = None):

    if sigmas is not None:
        two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
        return softmax(distances / two_sig_sq)
    else:
        return softmax(distances)

def binary_search(eval_fn, target, tol=1e-10, max_iter = 10000,
        lower = 1e-20, upper = 1000):

    """
    perform a binary search over input values to eval_fn

    # Arguments
    Eval_fn: function that we are optimising over
    target:  target value we want the function to output
    tol: Float, once our guess is close to the target, stop
    max_iter: Integer, maximum num. iterations to search for
    lower: float, lower bound of search range
    upper: float, upper bound of search range

    # Returns 
    Float, best input value to function found during search

    """

    for i in range(max_iter):
        guess = (lower + upper) / 2 
        val  = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess

        if np.abs(val - target) <= tol:
            break

    return guess

def calc_perplexity(prob_matrix):

    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    perplexity = 2 ** entropy
    return perplexity

def perplexity(distances, sigmas):
    return calc_perplexity(calc_prob_matrix(distances, sigmas))

def find_optimal_sigmas(distances, target_perplexity):
    """
    For each row of distances matrix, find simga that results in  target perplexity for that role
    """

    sigmas = []
    for i in range(distances.shape[0]):
        eval_fn = lambda sigma:perplexity(distances[i:i+1, :], np.array(sigma))

        correct_sigma = binary_search(eval_fn, target_perplexity)
        sigmas.append(correct_sigma)

    return np.array(sigmas)

def q_joint(Y):
    """
    Given the low-dimensional representation Y, compute the matrix of joint probabilities with entries q_ij
    """

    distances = neg_square_euc_distance(Y)
    exp_distances = np.exp(distances)
    np.fill_diagonal(exp_distances, 0)
    return exp_distances / np.sum(exp_distances), None


def p_conditional_to_joint(P):
    """
    given conditional probability matrix P, return approximiation of joint distribution probability
    """
    
    return (P + P.T) / (2. * P.shape[0])

def p_joint(X, target_perplexity):

    """
    Given a data matrix X, gives joint probabilities matrix

    #arguments
    X: Input data matrix
    # Returns
    P: Matrix with entries p_ij = joint probabilities
    """

    distances = neg_square_euc_distance(X)
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    p_conditional = calc_prob_matrix(distances, sigmas)
    P = p_conditional_to_joint(p_conditional)

    return P

def symmetric_sne_grad(P, Q, Y, _):
    """
    Estimate the gradient of  the cost with respect to Y
    """

    pq_diff = P - Q # N * N
    pq_expanded = np.expand_dims(pq_diff, 2) # N * N * 1
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0) # N * N * 2
    grad = 4. * (pq_expanded * y_diffs).sum(1) # N * 2
    return grad

def estimate_sne(X, y, P, rng, num_iters, q_fn, grad_fn, learning_rate, momentum, plot):

    """ 
    Estimate a SNE model

    # Arguments
    X: Input data matrix
    y: Class labels for that matrix
    P: Matrix of joint probabilities
    rng: np.random.RandomState().
    num_iters: Iterations to train for
    q_fn: Function that takes Y and gives Q prob matrix
    plot: How many times to plot during training

    # Returns
    Y: Matrix, low-dimensional representation of X

    """
    Y = rng.normal(0., 0.0001, [X.shape[0], 2])

    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()

    for i in range(num_iters):
        Q, distances = q_fn(Y)
        grads = grad_fn(P, Q, Y, distances)

        #Update Y
        Y = Y - learning_rate * grads

        if momentum:
            Y += momentum * (Y_m1 - Y_m2)
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()

        if plot and i %(num_iters / plot) == 0:
            categorical_scatter_2d(Y, y, alpha = 1., ms = 6, show = True, figsize = (9, 6))

    return Y

def q_tsne(Y):
    """
    t-SNE: Given low-dimensional representation Y, compute matrix of 
    joint probabilities with entries q_ij
    """

    distances = neg_square_euc_distance(Y)
    inv_distances = np.power(1. - distances, -1)
    np.fill_diagonal(inv_distances, 0.)
    return inv_distances / np.sum(inv_distances), inv_distances

def tsne_grad(P, Q, Y, inv_distances):
    """
    Estimate the gradient of t-SNE cost with respect to Y
    """

    pd_diff = P - Q
    pq_expanded = np.expand_dims(pd_diff, 2)
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
    distances_expanded = np.expand_dims(inv_distances, 2)

    y_diffs_wt = y_diffs * distances_expanded
    grad = 4 * (pq_expanded * y_diffs_wt).sum(1)

    return grad

def main():

    rng = np.random.RandomState(SEED)

    X, y = load_mnist('dataset/', digits_to_keep = CLASSES_TO_USE, N = NUM_POINTS)

    P = p_joint(X, PERPLEXITY)

    Y = estimate_sne(X, y, P, rng, 
            num_iters = NUM_ITERS,
            q_fn = q_tsne if TSNE else q_joint,
            grad_fn = tsne_grad if TSNE else symmetric_sne_grad,
            learning_rate = LEARNING_RATE,
            momentum = MOMENTUM,
            plot =  NUM_PLOTS)


    
if __name__ == "__main__":
    main()

