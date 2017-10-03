"""
 The following are a few elementary functions needed to implement the Deep ANN.
"""


"""
SET1
 The first set of functions return the said values along with a cache (activation_cache : Z).
 These are used for forward propagation
"""
def sigmoid(Z):
	A = np.sigmoid(Z)
	cache = Z
	return Z, cache

def relu(Z):
	A = np.maximum(Z,0)
	cache = Z
	return A, Z

def tanh(Z):
	A = (np.exp(Z) + np.exp(-Z) ) /(np.exp(Z) - np.exp(-Z))
    cache = Z
	return A, Z


"""
SET2
 These are the set of functions return the derivatives of the said functions.
 They are used in backward propagation.
"""
def sigmoid_backwards(dA, activation_cache):
    Z = activation_cache
	g_dashZ = np.multiply(sigmoid(Z), 1 - sigmoid(Z))
	dZ = np.multiply(Z, g_dashZ)
	return dZ

def relu_backwards(dA, activation_cache):
	Z = activation_cache
	g_dashZ = np.multiply(Z, Z>0)
	dZ = np.multiply(dA, g_dashZ)
	return dZ

def tanh_backwards(dA, activation_cache):
	Z = activation_cache
	g_dashZ = 1 - (tanh(Z)**2)
	dZ = np.multiply(dA, g_dashZ)

	return dZ


"""
SET3
 These functions compute the cost and return the same.
"""


def compute_cost(AL, Y, parameters, lambd = 0):

	m = Y.shape[1]
	L = len(parameters)//2

	cost = -(1/m)*(np.dot(Y , np.log(AL).T) + np.dot(1-Y, np.log(1-AL).T) )
	cost = np.squeeze(cost)

	if lambd != 0:
		L2_regularization_cost = 0
		for l in xrange(L):
			L2_regularization_cost +=  np.sum(np.square(parameters["W" + str(l+1)]))

		L2_regularization_cost *= (lambd/(2*m))
		cost += L2_regularization_cost

	return cost
