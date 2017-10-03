

def linear_backward(dZ, cache):

	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = (1/m)*np.dot(dZ,A_prev.T)
	db = (1/m)*np.sum(dZ, axis = 1, keepdims = True)
	dA_prev = np.dot(W.T,dZ)

	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):

	linear_cache , activation_cache = cache

	if activation  == "sigmoid" :
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev , dW, db = linear_backwards(dZ, linear_cache)

	if activation == "relu" :
		dZ = relu_backward(dA, activation_cache)
		dA_prev , dW, db = linear_backwards(dZ, linear_cache)

	if activation == "tanh" :
		dZ = tanh_backwards(dA, activation_cache)
		dA_prev, dW, db = linear_backwards(dZ, linear_cache)

	return dA_prev , dW, db


def L_model_backwards(AL, Y, caches, lambd = 0, keep_prob = 1.0):

	grads = {}
	L = len(caches)
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	dAL = -(np.divide(Y,AL) - np.divide(1-Y, 1-AL) )

	current_cache = caches[L-1]
	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

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
