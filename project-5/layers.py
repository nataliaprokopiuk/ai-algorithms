import torch
from activation_func import ActivationFunc as act

class Layer:
	def __init__(self):
		self.input_data = None
		self.linear_output = None
		self.output_data = None

	def forward_propagation(self, input_data):
		raise NotImplementedError

	def backward_propagation(self, error, learning_rate, momentum: float):
		raise NotImplementedError
	

class DenseLayer(Layer):
	def __init__(self, input_size: int, output_size: int, activation: str, activation_prime):
		super().__init__()
		self.weights = torch.rand(input_size, output_size) - 0.5
		self.bias = torch.rand(1, output_size) - 0.5
		self.activation = activation
		self.activation_prime = activation_prime
		self.velocity_w = torch.zeros_like(self.weights)
		self.velocity_b = torch.zeros_like(self.bias)

		act_dict = {
			'identity': (act.identity, act.identity_prime),
			'relu': (act.relu, act.relu_prime),
			'leaky_relu': (act.leaky_relu, act.leaky_relu_prime),
			'sigmoid': (act.sigmoid, act.sigmoid_prime),
		}

		if activation in act_dict.keys():
			self.activation, self.activation_prime = act_dict[activation]
		else:
			raise ValueError(f'Unknown activation function: {activation}')

	def forward_propagation(self, input_data):
		self.input_data = input_data
		self.linear_output = torch.matmul(input_data, self.weights) + self.bias
		self.output_data = self.activation(self.linear_output)
		return self.output_data
	
	def backward_propagation(self, output_error, learning_rate, momentum: float):
		activation_error = self.activation_prime(self.linear_output) * output_error
		input_error = torch.matmul(activation_error, self.weights.T)
		weights_error = torch.matmul(self.input_data.T, activation_error)
		bias_error = torch.sum(activation_error, axis=0, keepdim=True)

		if momentum:
			self.velocity_w = momentum * self.velocity_w + learning_rate * weights_error
			self.velocity_b = momentum * self.velocity_b + learning_rate * bias_error
			self.weights -= self.velocity_w
			self.bias -= self.velocity_b
		else:
			self.weights -= learning_rate * weights_error
			self.bias -= learning_rate * bias_error
		return input_error