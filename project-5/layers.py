import torch

class Layer:
	def __init__(self):
		self.input_data = None
		self.output_data = None

	def forward_propagation(self, input_data):
		raise NotImplementedError

	def backward_propagation(self, error, learning_rate, momentum: float):
		raise NotImplementedError
	

class DenseLayer(Layer):
	def __init__(self, input_size: int, output_size: int, activation, activation_derivative):
		self.weights = torch.rand(input_size, output_size) - 0.5
		self.bias = torch.rand(1, output_size) - 0.5
		self.activation = activation
		self.activation_derivative = activation_derivative
		self.velocity = 0

	def forward_propagation(self, input_data):
		self.input_data = input_data
		self.output_data = torch.matmul(input_data, self.weights) + self.bias
		self.activated_output = self.activation(self.output_data)
		return self.activated_output
	
	def backward_propagation(self, output_error, learning_rate, momentum: float):
		input_error = torch.matmul(output_error, self.weights.T)
		weights_error = torch.matmul(self.input_data.T, output_error)

		activation_error = output_error * self.activation_derivative(self.output)
		input_error = np.dot(activation_error, self.weights.T)
		weights_error = np.dot(self.input.T, activation_error)

		# Update parameters
		self.weights -= learning_rate * weights_error
		self.bias -= learning_rate * activation_error
		return input_error