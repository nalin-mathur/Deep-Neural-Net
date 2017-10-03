"""
FUNCTIONS NEEDED FOR BACKWARD PROPAGATION
"""

import numpy as np
from elementary_functions import *

def linear_backward(dZ, cache):
    """
    Implements linear part of backward propogation.

    Input:
        dZ : The derivative with respect to linear activation values of given layer.
        cache : linear cache values for calculation of derivatives.
            [A_prev, W, b, (D) ] . D is present only when dropout is implemented.

    Output:
        dA_prev : Numpy matrix containing derivatives with respect to activations of previous layer.
        dW : Numpy matrix containing derivatives with respect to weights of current layer.
        db : Numpy matrix containing derivatives with respect to biases of current layer.
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ,cache[0].T)
    db = (1/m)*np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(cache[1].T,dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Carries out backward propagation for one layer.

    Input:
        dA : Derivative with respect to the activations of this layer.
        cache : Cache associated with this particular layer.
        activation : Activation funciton of given layer.

    Output:
        dA_prev : Numpy matrix containing derivatives with respect to activation of previous layer.
        dW : Numpy matrix containing derivatives with respect to weights of current layer.
        db : Numpy matrix containing derivatives with respect to biases of current layer.
    """

    linear_cache , activation_cache = cache

    if activation  == 'sigmoid' :
    	dZ = sigmoid_backwards(dA, activation_cache)
    	dA_prev , dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'relu':
		dZ = relu_backwards(dA, activation_cache)
		dA_prev , dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'tanh':
		dZ = tanh_backwards(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)


    return dA_prev , dW, db


def L_model_backwards(AL, Y, caches, lambd = 0, keep_prob = 1.0):
    """
    Implements backward pass through the NN in accordance to the caches and final values calculated.

    Input:
        AL : Numpy matrix with the activations/output of the final layer calculated using our ANN. Dimensions : (#classes, #examples).
        Y : Numpy matrix with the actual output of the given examples. Dimensions : (#classes, #examples).
        caches : Python list containing caches of every layer in the ANN.
        lambd : Regularisation factor.
        keep_prob : Dropout factor.

    Output:
        grads : Python dictionary containing all the required gradients.
    """

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    grads['dA' + str(L)] = -(np.divide(Y,AL + 1e-10) - np.divide(1-Y, 1-AL + 1e-10) )

    current_cache = caches[L-1]

    for l in reversed(range(L-1)):

    	current_cache = caches[l]
    	linear_cache, activation_cache = current_cache
    	W = linear_cache[1]
    	grads["dA" + str(l+1)], grads["dW" + str(l+1)], grads["db" + str(l+1)] = linear_activation_backward(grads["dA" + str(l+2)], current_cache, 'relu')

    	if lambd != 0:
    		grads["dW" + str(l+1)] += (lambd/m)*W

    	if keep_prob < 1.0:
    		D = linear_cache[3]
    		grads["dA" + str(l+1)] *= D
    		grads["dA" + str(l+1)] /= keep_prob

    return grads
