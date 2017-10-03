import numpy as np
import matplotlib.pyplot as plt

"""
 The following are a few elementary functions needed to implement the Deep ANN :

"""
def sigmoid(Z):
	A = np.sigmoig(Z)
	cache = Z
	return Z, cache

def relu(Z):
	A = np.maximum(Z,0)
	cache = Z
	return A, Z

def tanh(Z):
	cache = Z
	A = (np.exp(Z) + np.exp(-Z) ) /(np.exp(Z) - np.exp(-Z))
	return A, Z

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
End of elementary functions
"""



def random_mini_batches(X, Y, mini_batch_size = 64):
	"""
	Randomly generates mini batches from given data

	Input :
		X : Numpy matrix containing input data. Dimensions : (number of features in each example, number of examples)
		Y : Numpy matrix containing labels. Dimensions : (1, number of examples)
		mini_batch_size : Size of a single mini batch that is to be made.

	Output :
		Python list containing all the different randomly generated mini batches
	"""


	m = X.shape[1]
	mini_batches = []

	#Shuffling around the data randomly according to the 'permutation' list
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((1,m))

	complete_minibatch_number = math.floor(m/mini_batch_size)

	for k in xrange(complete_minibatch_number):

		mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]

		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	if m%mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, (k+1)*mini_batch_size : m]
		mini_batch_Y = shuffled_Y[:, (k+1)*mini_batch_size : m]

		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	return mini_batches


def initialize_parameters(layer_dims, initialization_factor = '0.01'):

	"""
	Initializes the parameters of our neural network

	Input :
		layer_dims : Python list containing lengths of all the layers of the ANN"
		initialization_factor : string containing type of initialization facotor required.

	Output :
		Python dictionary containing all the randomly initialized parameters "W1", "b1" , "W2" , "b2"...
		Wl -- weight matrix between the layers l-1 and l of NN with dimensions (layers_dims[l-1],layers_dims[l])
		b1 -- bias vector of layer l with dimensions (layers[l],1)
	"""

	parameters = {}
	depth = len(layer_dims)

	for l in xrange(1,depth):

		if initialization_factor == '0.01':
			parameters['W' + str(l)] = np.random.randn(layer_dims[l] , layer_dims[l-1])*0.01
		if initialization_factor == 'he':
			parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
		if initialization_factor == 'xavier':
			parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(1/layer_dims[l-1])

		parameters['b' + str(l)] = np.zeros( (layer_dims[l], 1) )*0.01

	return parameters

def initialize_momentum_velocity(parameters):
	"""
	Initializes the vdW and vdb parameters for momentum optimizer.

	Input:
		parameters : Python dictionary containing all the randomly initialized weights and biases for each layer.
	Output:
		v : Python dictionary containing the momentums initialized to zero.
	"""

	L = len(parameters)//2
	v = {}

	for l in xrange(L):
		v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
		v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)

	return v

def initialize_adam(parameters):
	"""
	Initializes the vdW, vdb, sdW and sdb parameters for adams optimizer.

	Input:
		parameters : Python dictionary containing all the randomly initialized weights and biases for each layer.
	Output:
		v : Python dictionary containing the momentum terms initialized to zero.
		s : Python dictionary containing the RMSprop terms initialized to zero.
	"""

	v = initialize_momentum_velocity(parameters)
	s = initialize_momentum_velocity(parameters)

	return v, s



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

def update_parameters_gradient_descent(parameters, grads, learning_rate):

	L = len(parameters)//2

	for l in range(1,L+1):
		parameters["W" + str(l)] -= learning_rate*grads["dW" + str(l)]
		parameters["b" + str(l)] -= learning_rate*grads["db" + str(l)]

	return parameters

def update_parameters_momentum(parameters, grads, v, beta, learning_rate):

	L = len(parameters)//L

	for l in xrange(1,L+1):
		v["dW" + str(l)] = beta*v["dW" + str(l)] + (1-beta)*grads["dW" + str(l)]
		v["db" + str(l)] = beta*v["db" + str(l)] + (1-beta)*grads["db" + str(l)]

		parameters["W" + str(l)] -= learning_rate*v["dW" + str(l)]
		parameters["b" + str(l)] -= learning_rate*v["dW" + str(l)]

	return parameters, v

def update_parameters_adam(parameters, grads, v, s, t,learning_rate, beta1 = 0.9, beta2 = 0.999 , epsilon = 1e-8):

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


def nn_model(X, Y, layer_dims, learning_rate = 0.0001, num_epochs = 12000, optimizer = 'gradient_descent', lambd = 0, keep_prob = 1.0, mini_batch_size = 64, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, print_cost = True):

	"""
	Entire Nerual Network Model which returns learned parameters.

	Input:
		X : Training set examples used to train the ANN. Dimensions : (number of features per example, number of examples). Every column acts as one example.
		Y : Lables for the above mentioned training set. Dimensions : (1 , number of examples).
		layer_dims : List containing length of layers of an ANN starting from the first to the last layer.
		learning_rate : The rate by which you want your algorithm to learn.
		num_epochs : Number of iterations to be conducted for learning.
		optimizer : Choice of optimizer to be chosen to optimize the ANN. Choices : 'gradient_descent' , 'momentum' , 'Adams'
		lambd : L2 Regularisation hyperparameterTo be changed when we want to change rate of regularisation.
		keep_prob : Number between 0 and 1. Parameter dictating probability of chosing an output when implementing dropout regularisation.
		mini_batch_size : integer that dictates the number of examples to be considered in a mini batch when implementing mini batch descent.
		beta1 : Number between 0 and 1. Serves as beta when implementing momentum optimizer and as beta1 when implementing Adams optimization.
		beta2 : Number between 0 and 1. Serves as beta2 during implementation of Adams optimization.
		epsilon : Parameter for adams optimization.
		print_cost : True if you want to print the cost for every 1000th iteration.

	Output:
		parameters : Python dictionary containing all the learnt parameters of the ANN.

	"""

	L = len(layer_dims)
	costs = []
	t = 0

	parameters = initialize_parameters(layer_dims)

	if optimizer == 'momentum':
		v = initialize_momentum_velocity(parameters)
	elif optimizer == 'Adams':
		v, s = initialize_adam(parameters)

	for i in xrange(num_epochs):

		mini_batches = random_mini_batches(X, Y, mini_batch_size)

		for mini_batch in mini_batches:

			mini_batch_X , mini_batch_Y = mini_batch

			yhat, caches = L_model_forward(mini_batch_X, parameters, keep_prob)

			cost = compute_cost(yhat, mini_batch_Y, parameters, lamb)

			grads = L_model_backwards(yhat, mini_batch_Y, caches, lambd, keep_prob)

			if optimizer == 'gradient_descent':
				parameters = update_parameters_gradient_descent(parameters, grads, learning_rate)
			elif optimizer == 'momentum':
				parameters, v = update_parameters_momentum(parameters, grads, v, beta1, learning_rate)
			elif optimizer == 'Adams':
				t += 1
				parameters, v, s = update_parameters_adam(parameters, grads, v, s, t, beta1 , beta2  , epsilon , learning_rate)


		if print_cost and i%1000 == 0:
			print "Cose after after epoch " , i , " : " , cost
		if i%100 == 0:
			costs.append(cost)

	plt.plot(costs)
	plt.ylabel('Costs')
	plt.xlabel('Epochs (every 100 epochs)')
	plt.title('Learning rate = ' + str(learning_rate))
	plt.show()

	return parameters
