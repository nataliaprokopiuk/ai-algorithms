import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_function, num_hidden_layers, dropout_rate=0.3):
        super(MLP, self).__init__() 
        self.num_hidden_layers = num_hidden_layers

        # input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        activation_functions = {
        "identity": nn.Identity(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
        "sigmoid": nn.Sigmoid()
        }
        self.activation_function = activation_functions[activation_function]

        # hidden layers with chosen activation function
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num_hidden_layers)])
        self.dropout = nn.Dropout(dropout_rate)

        # output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        x = self.input_layer(input)

        # Apply activation function
        x = self.activation_function(x)
        x = self.dropout(x)
        
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.activation_function(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x
