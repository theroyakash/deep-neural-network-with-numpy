import numpy as np

class InitializeParameters():

	def __init__(self, layer_dims):
		self.layer_dims = layer_dims

	def initialize(self):
		"""
		Arguments:
		layer_dims -- python array (list) containing the dimensions of each layer in our network
		
		Returns:
		parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
						Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
						bl -- bias vector of shape (layer_dims[l], 1)
		"""
		
		parameters = {}
		L = len(self.layer_dims)  # number of layers in the network

		for l in range(1, L):
			parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
			parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
			
			assert(parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l-1]))
			assert(parameters['b' + str(l)].shape == (self.layer_dims[l], 1))

		return parameters