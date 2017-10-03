"""
FUNCTIONS FOR PARAMETER UPDATING
"""

def update_parameters_gradient_descent(parameters, grads, learning_rate):
    """
    Updates parameters by gradient descent algorithm.

    Input:
        parameters : Python dictionary containing weights and biases.
        grads : Python dictionary containing all gradients.
        learning_rate : For rate of descent.

    Output:
        parameters : Updated parameters dictionary.
    """

	L = len(parameters)//2

	for l in range(1,L+1):
		parameters["W" + str(l)] -= learning_rate*grads["dW" + str(l)]
		parameters["b" + str(l)] -= learning_rate*grads["db" + str(l)]

	return parameters

def update_parameters_momentum(parameters, grads, v, beta, learning_rate):
    """
    Updates parameters by momentum algorithm.

    Input:
        parameters : Python dictionary containing weights and biases.
        grads : Python dictionary containing all gradients.
        v : Python dictionary containing all momentum/velocity terms.
        beta : Constant for momentum updating.
        learning_rate : For rate of descent.

    Output:
        parameters : Updated parameters dictionary.
        v : Updated moments/velocities.
    """

	L = len(parameters)//L

	for l in xrange(1,L+1):
		v["dW" + str(l)] = beta*v["dW" + str(l)] + (1-beta)*grads["dW" + str(l)]
		v["db" + str(l)] = beta*v["db" + str(l)] + (1-beta)*grads["db" + str(l)]

		parameters["W" + str(l)] -= learning_rate*v["dW" + str(l)]
		parameters["b" + str(l)] -= learning_rate*v["dW" + str(l)]

	return parameters, v


def update_parameters_adam(parameters, grads, v, s, t,learning_rate, beta1 = 0.9, beta2 = 0.999 , epsilon = 1e-8):

    """
    Updates parameters by momentum algorithm.

    Input:
        parameters : Python dictionary containing weights and biases.
        grads : Python dictionary containing all gradients.
        v : Python dictionary containing all momentum/velocity terms.
        s : Python dictionary containing all RMSprop terms.
        t : Number which tells us the pass. For correction of v and s terms.
        learning_rate : For rate of descent.
        beta1 : Constant for momentum updating.
        beta2 : Constant for RMSprop updating.
        epsilon: For correction of correction. LOL.

    Output:
        parameters : Updated parameters dictionary.
        v : Updated moments/velocities.
        s : Updated RMSprop terms.

    """

	L = len(parameters)//2
	v_corrected = {}
	s_corrected = {}

	for l in xrange(1,L+1):
		v["dW" + str(l)] = beta1*v["dW" + str(l)] + (1-beta1)*grads["dW" + str(l)]
		v["db" + str(l)] = beta1*v["db" + str(l)] + (1-beta1)*grads["db" + str(l)]

		v_corrected["dW" + str(l)] = v["dW" + str(l)]/(1-(beta1**t))
		v_corrected["db" + str(l)] = v["dW" + str(l)]/(1-(beta1))

		s["dW" + str(l)] = beta2*v["dW" + str(l)] + (1-beta2)*(grads["dW" + str(l)]**2)
		s["db" + str(l)] = beta2*v["db" + str(l)] + (1-beta2)*(grads["db" + str(l)]**2)

		s_corrected["dW" + str(l)] = s["dW" + str(l)]/(1-(beta2**t))
		s_corrected["db" + str(l)] = s["db" + str(l)]/(1-(beta2**t))

		grads["dW" + str(l)] -= learning_rate*(v_corrected["dW" + str(l)]/np.sqrt(s_corrected["dW" + str(l)] + epsilon))
		grads["db" + str(l)] -= learning_rate*(v_corrected["db" + str(l)]/np.sqrt(s_corrected["db" + str(l)] + epsilon))


	return parameters, v, s
