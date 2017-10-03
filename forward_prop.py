"""
FUNCTIONS NEEDED FOR FORWARD PROPAGATION
"""
def linear_forward(A_prev , W , b):

	"""
	Implements linear part of the NN forward propogation

	Input:
		A_ : Activations  matrix of the previous layer which serves as input to the present layer. Dimensions : (length of previous layer, number of examples)
		W : Weight matrix for current layer. Dimensions : (length of current layer, length of previous layer)
		b : Bias vector for current layer. Dimensions : (length of current layer, 1)

	Output:
		Z : Input of the activation function of the current layer or the 'preactivation parameter'.
		cache : Python dictionary containing 'A', 'W', 'b' ; stored for computing the backward pass effeciently.

	"""
	Z = np.dot(W,A_prev) + b
	cache = (A_prev,W,b)

	return Z , cache



def linear_activation_forward(A_prev, W, b , activation = "relu"):

	"""
	Implements the forward propogation for Linear->Activation Layer

	Input:
		A_prev : Activation matrix from the previous layer. Dimensions: (length of previous layer, number of examples)
		W : Weight matrix for current layer. Dimensions : (length of current layer, length of previous layer)
		b : Bias matrix for current layer. Dimensions : (length of current layer, 1)
		activation : string that tells the function the activation to be used in the layer. Options for activation : "relu", "sigmoid", "tanh".

	Output:
		A : Activation of the current layer. 'Post Activation values'. Dimensions : (length of current layer, number of examples)
		cache : python dictionary containing "linear cache" and "activation cache". Used for the backward pass.
			"linear cache" : Output cache of 'linear_forward' function
			"activation cache" : Z
				"""

	if activation == 'sigmoid':
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)

	if activation ==  'relu':
		Z, linear_cace = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)
	if activation == 'tanh' :
		Z, linear_cache = linear_forward(A_prev,W,b)
		A, activation_cache = tanh(Z)

	cache = (linear_cache, activation_cache)

	return A, cache



def L_model_forward(X, parameters, keep_prob = 1.0):

	"""
	Implements forward pass through the NN according to the activations given.

	Input:
		X: Data Matrix. Dimensions : (number of features/input size, number of examples)
		parameters: Weight and Bias parameters of the NN. Initialized randomly using 'initialize_parameters' funciton.
		keep_prob : Probability with which to eliminate nodes. To be used only when using dropout regularization.

	Output:
		yhat: Post activation values of last layer.
		caches: list of all caches in forward pass.
			total number of caches : L-1 (caches of all layers except the last; L = number of layers of NN) indexed from 0 to L-2
			every element has 'linear_cache' and 'activation_cache' as output by 'linear_activation_forward' function.

	"""

	caches = []
	A = X
	L = len(parameters)//2  # Number of layers of the neural network. Each layer has one parameter of weights and one of the biases.

	for l in xrange(1,L-1):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)] , parameters['b' + str(l)], 'relu')

		if keep_prob < 1.0:
			D = np.random.randn(A.shape[0], A.shape[1])
			D = D < keep_prob
			A = np.multiply(A,D)
			A /= keep_prob
			linear_cache, activation_cache = cache
			linear_cache.append(D)
			cache = (linear_cache, activation_cache)

		caches.append(cache)

	yhat, cache = linear_activation_forward(A, parameters['W' + str(L)] , parameters['b' + str(L)], 'sigmoid' )
	caches.append(cache)

	return yhat, caches
