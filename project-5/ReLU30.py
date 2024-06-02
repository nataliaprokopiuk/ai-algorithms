import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from perceptron import MLP
from manage_data import prepare_dataset

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for input, decision in train_loader:
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, decision)
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item() * input.size(0)  # Accumulate loss

    return running_loss / len(train_loader.dataset)  # Return average loss

if __name__ == "__main__":
    # Define the number of hidden layers and activation function
    num_hidden_layers = 30
    activation_function = nn.ReLU()

    input_size = 16  # Example input size, adjust as needed
    hidden_size = 128  # Example hidden layer size, adjust as needed
    output_size = 4  # Example output size, adjust as needed

    # Prepare the dataset
    train_dataset, validation_dataset, test_dataset = prepare_dataset()

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

    # Initialize the model with the correct number of hidden layers
    model = MLP(input_size, hidden_size, output_size,
                activation_function=activation_function, num_hidden_layers=num_hidden_layers)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6)

    # Train the model for one epoch to get gradients
    train_loss = train_model(model, train_loader, criterion, optimizer)

    # Calculate gradient norms for each weight matrix
    gradient_norms = []
    layer_count = 0
    weight_norms = []
    bias_norms = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'weight' in name:
                weight_norms.append(param.grad.norm().item())
            elif 'bias' in name:
                bias_norms.append(param.grad.norm().item())

    # Average the gradient norms for each layer
    for i in range(num_hidden_layers + 1):
        weight_norm = weight_norms[i]
        bias_norm = bias_norms[i]
        avg_norm = (weight_norm + bias_norm) / 2
        gradient_norms.append(avg_norm)

    # Print the gradient norms for each layer
    for i, norm in enumerate(gradient_norms):
        print(f"Average gradient norm for layer {i+1}: {norm}")

    # Calculate and print the overall average gradient norm
    average_gradient_norm = sum(gradient_norms) / len(gradient_norms)
    print(f"Overall average gradient norm during the first epoch: {average_gradient_norm}")
