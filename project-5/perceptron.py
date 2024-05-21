import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=5, weigths=None): 
        super(MLP, self).__init__() 
        self.num_hidden_layers = num_hidden_layers

        if weigths:
            with torch.no_grad():
                self.linear.weight.copy_(weigths)

        # input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # hidden layers with chosen activation function
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num_hidden_layers)])

        # output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input): 
        x = self.input_layer(input)
        x = nn.functional.relu(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = nn.functional.relu(x)
        x = self.output_layer(x)
        return x