import numpy as np
from matplotlib import pyplot as plt
from activations import relu, sigmoid
from initializeparameters import InitializeParameters


class NeuralNetwork():

	def linear_forward(self, A, W, b):
		"""
		Implements a linear forward step
			Args:
				- A: activations from previous layer (or input data): (size of previous layer, number of examples)
				- W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
				- b: bias vector, numpy array of shape (size of the current layer, 1)

			Returns:
				- Z: the input of the activation function, also called pre-activation parameter cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently

		"""

		Z = W.dot(A) + b

		assert(Z.shape == (W.shape[0], A.shape[1]))
		cache = (A, W, b)

		return Z, cache

	# Linear activation forward

	def linear_activation_forward(self, A_prev, W, b, activation):
		"""
		Args:
			- A_prev: activations from previous layer (or input data): (size of previous layer, number of examples)
			- W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
			- b: bias vector, numpy array of shape (size of the current layer, 1)
			- activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

		Returns:
			- A: The output of the activation function, also called the post-activation value
			- cache: a python tuple containing "linear_cache" and "activation_cache"; stored for computing the backward pass efficiently
		"""

		if activation == "sigmoid":
			# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
			Z, linear_cache = self.linear_forward(A_prev, W, b)
			A, activation_cache = sigmoid(Z)

		elif activation == "relu":
			# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
			Z, linear_cache = self.linear_forward(A_prev, W, b)
			A, activation_cache = relu(Z)

		assert (A.shape == (W.shape[0], A_prev.shape[1]))
		cache = (linear_cache, activation_cache)

		return A, cache

	def L_model_forward(self, X, parameters):
		"""
		Args:
			- X -- data, numpy array of shape (input size, number of examples)
			- parameters -- output of initialize_parameters_deep()

		Returns:
			- AL -- last post-activation value
			- caches -- list of caches containing every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
		"""

		caches = []
		A = X
		L = len(parameters) // 2   # number of layers in the neural network

		for l in range(1, L):
			A_prev = A
			W = parameters["W" + str(l)]
			b = parameters["b" + str(l)]
			A, cache = self.linear_activation_forward(A_prev, W, b, "relu")
			caches.append(cache)

		W , b = parameters["W"+str(L)], parameters["b"+str(L)]
		AL, cache = self.linear_activation_forward(A, W, b, "sigmoid")
		caches.append(cache)

		assert(AL.shape == (1,X.shape[1]))

		return AL, caches

	# Cost computation:
	def compute_cost(self, AL, Y):
		"""
		Args:
			- AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
			- Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

		Returns:
			- cost -- cross-entropy cost
		"""

		m = Y.shape[1]

		# Compute loss from aL and y.
		cost = -1/m * np.sum(Y*(np.log(AL)) + (1-Y)*(np.log(1-AL)))

		# To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
		cost = np.squeeze(cost)

		assert(cost.shape == ())

		return cost


	# Back propagation:

	def linear_backward(self, dZ, cache):
		"""
		Args:
			- dZ -- Gradient of the cost with respect to the linear output (of current layer l)
			- cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

		Returns:
			- dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
			- dW -- Gradient of the cost with respect to W (current layer l), same shape as W
			- db -- Gradient of the cost with respect to b (current layer l), same shape as b
		"""
		A_prev, W, b = cache
		m = A_prev.shape[1]

		dW = (1.0/m) * dZ.dot(A_prev.T)
		db = (1.0/m) * np.sum(dZ,axis=1,keepdims=True)
		dA_prev = W.T.dot(dZ)

		assert (dA_prev.shape == A_prev.shape)
		assert (dW.shape == W.shape)
		assert (db.shape == b.shape)

		return dA_prev, dW, db

	# Linear activation backward

	def relu_backward(self, dA, cache):
		"""
		Implements the backward propagation for a single RELU unit.

			Args:
				- dA -- post-activation gradient, of any shape
				- cache -- 'Z' where we store for computing backward propagation efficiently

			Returns:
				- dZ -- Gradient of the cost with respect to Z
		"""

		Z = cache
		dZ = np.array(dA, copy=True) # just converting dz to a correct object.

		# When z <= 0, you should set dz to 0 as well.
		dZ[Z <= 0] = 0
		assert (dZ.shape == Z.shape)

		return dZ

	def sigmoid_backward(self, dA, cache):
		"""
		Implement the backward propagation for a single SIGMOID unit.

			Arguments:
				- dA -- post-activation gradient, of any shape
				- cache -- 'Z' where we store for computing backward propagation efficiently

			Returns:
				- dZ -- Gradient of the cost with respect to Z
		"""

		Z = cache
		s = 1/(1+np.exp(-Z))
		dZ = dA * s * (1-s)
		assert (dZ.shape == Z.shape)

		return dZ

	def linear_activation_backward(self, dA, cache, activation):
		"""
		Back propagation unit
			Arguments:
				- dA -- post-activation gradient for current layer l
				- cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
				- activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

			Returns:
				- dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
				- dW -- Gradient of the cost with respect to W (current layer l), same shape as W
				- db -- Gradient of the cost with respect to b (current layer l), same shape as b
		"""
		linear_cache, activation_cache = cache

		if activation == "relu":
			dZ = self.relu_backward(dA, activation_cache)
			dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

		elif activation == "sigmoid":
			dZ = self.sigmoid_backward(dA, activation_cache)
			dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

		return dA_prev, dW, db

	# Implementation of the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID

	def L_model_backward(self, AL, Y, caches):
		"""
		Arguments:
		AL -- probability vector, output of the forward propagation (L_model_forward())
		Y -- true "label" vector
		caches -- list of caches containing:
		every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
		the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

		Returns:
		grads -- A dictionary with the gradients
		grads["dA" + str(l)] = ...
		grads["dW" + str(l)] = ...
		grads["db" + str(l)] = ...
		"""
		grads = {}
		L = len(caches) # the number of layers
		m = AL.shape[1]
		Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

		# Initializing the backpropagation
		dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

		# Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
		current_cache = caches[L-1]
		grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")

		# Loop from l=L-2 to l=0
		for l in reversed(range(L-1)):
			# lth layer: (RELU -> LINEAR) gradients.
			# Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
			current_cache = caches[l]
			dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA"+str(l+1)], current_cache, "relu")
			grads["dA" + str(l)] = dA_prev_temp
			grads["dW" + str(l + 1)] = dW_temp
			grads["db" + str(l + 1)] = db_temp

		return grads


	# Update Parameters function using gradient descent

	def update_parameters(self, parameters, grads, learning_rate):
		"""
		Arguments:
		parameters -- python dictionary containing your parameters
		grads -- python dictionary containing your gradients, output of L_model_backward

		Returns:
		parameters -- python dictionary containing your updated parameters
		parameters["W" + str(l)] = ...
		parameters["b" + str(l)] = ...
		"""

		L = len(parameters) // 2 # number of layers in the neural network

		# Update rule for each parameter. Use a for loop.

		for l in range(L):
			parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l + 1)]
			parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l + 1)]

		return parameters

	def L_layer_model(self, X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False): #lr was 0.009
		"""
		Arguments:
		X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
		Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
		layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
		learning_rate -- learning rate of the gradient descent update rule
		num_iterations -- number of iterations of the optimization loop
		print_cost -- if True, it prints the cost every 100 steps

		Returns:
		parameters -- parameters learnt by the model. They can then be used to predict.
		"""

		costs = []   # keep track of cost

		# Parameters initialization
		initializer = InitializeParameters(layers_dims)
		parameters = initializer.initialize()

		# Loop (gradient descent)
		for i in range(0, num_iterations):

			# Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
			AL, caches = self.L_model_forward(X, parameters)

			# Compute cost.
			cost = self.compute_cost(AL, Y)

			# Backward propagation.
			grads = self.L_model_backward(AL, Y, caches)

			# Update parameters.
			parameters = self.update_parameters(parameters, grads, learning_rate)

			# Print the cost every 100 training example
			if print_cost and i % 100 == 0:
				print ("Cost after iteration %i: %f" %(i, cost))
			if print_cost and i % 100 == 0:
				costs.append(cost)

		# plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per hundreds)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()

		return parameters

	def predict(self, X, y, parameters):
		"""
		This function is used to predict the results of a  L-layer neural network.

		Arguments:
		X -- data set of examples you would like to label
		parameters -- parameters of the trained model

		Returns:
		p -- predictions for the given dataset X
		"""

		m = X.shape[1]
		n = len(parameters) // 2 # number of layers in the neural network
		p = np.zeros((1,m))

		# Forward propagation
		probas, caches = self.L_model_forward(X, parameters)


		# convert probas to 0/1 predictions
		for i in range(0, probas.shape[1]):
			if probas[0,i] > 0.5:
				p[0,i] = 1
			else:
				p[0,i] = 0

		print("Accuracy: "  + str(np.sum((p == y)/m)))

		return p
