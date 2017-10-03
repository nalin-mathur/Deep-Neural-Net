

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
