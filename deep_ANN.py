import numpy as np
import matplotlib.pyplot as plt
from elementary_functions import *
from initializer_functions import *
from forward_prop import *
from backward_prop import *
from update_functions import *


def nn_model(X, Y, layer_dims, learning_rate = 0.0001, num_epochs = 15000, optimizer = 'gradient_descent', initialization_factor = '0.01', lambd = 0, keep_prob = 1.0, mini_batch_size = 64, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, print_cost = True):

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

	parameters = initialize_parameters(layer_dims, initialization_factor)

	if optimizer == 'momentum':
		v = initialize_momentum_velocity(parameters)
	elif optimizer == 'Adams':
		v, s = initialize_adam(parameters)

	# yhat, accuracy = predict(parameters, X, Y)
	# print "The accuracy is : " , accuracy

	for i in xrange(num_epochs):

		mini_batches = random_mini_batches(X, Y, mini_batch_size)

		for mini_batch in mini_batches:

			mini_batch_X , mini_batch_Y = mini_batch

			yhat, caches = L_model_forward(mini_batch_X, parameters, keep_prob)

			cost = compute_cost(yhat, mini_batch_Y, parameters, lambd)
		
			grads = L_model_backwards(yhat, mini_batch_Y, caches, lambd, keep_prob)

			if optimizer == 'gradient_descent':
				parameters = update_parameters_gradient_descent(parameters, grads, learning_rate)
			elif optimizer == 'momentum':
				parameters, v = update_parameters_momentum(parameters, grads, v, beta1, learning_rate)
			elif optimizer == 'Adams':
				t += 1
				parameters, v, s = update_parameters_adam(parameters, grads, v, s, t, beta1 , beta2  , epsilon , learning_rate)


		if print_cost and i%1000 == 0:
			print "Cost after after epoch " , i , " : " , cost
		if i%100 == 0:
			costs.append(cost)

	plt.plot(costs)
	plt.ylabel('Costs')
	plt.xlabel('Epochs (every 100 epochs)')
	plt.title('Learning rate = ' + str(learning_rate))
	plt.show()

	return parameters

def predict(X, parameters, Y, keep_prob = 1.0):

	yhat , caches = L_model_forward(X, parameters, keep_prob)
	accuracy = np.sum(yhat==Y)/(Y.shape[1])
	return yhat , accuracy















