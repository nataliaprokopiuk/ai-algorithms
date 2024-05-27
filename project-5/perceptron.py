import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_function, num_hidden_layers):
        super(MLP, self).__init__() 
        self.num_hidden_layers = num_hidden_layers

        # input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.activation_function = activation_function

        # hidden layers with chosen activation function
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num_hidden_layers)])

        # output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        x = self.input_layer(input)

        # Apply activation function
        x = self.activation_function(x)
        
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.activation_function(x)
        x = self.output_layer(x)
        return x
