import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from perceptron import MLP
from manage_data import prepare_dataset
import os
import pandas as pd


def validate_model(model_path, activation_function, num_hidden_layers):
    # Load datasets
    _, _, test_dataset = prepare_dataset()

    # Data loader
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Load the model
    model = MLP(16, 32, 4, activation_function=activation_function, num_hidden_layers=num_hidden_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Lists to store true labels and predicted labels
    y_true = []
    y_pred = []

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Validation phase
    test_loss = 0.0

    # Iterate over test dataset and make predictions
    with torch.no_grad():
        for input, decision in test_loader:
            output = model(input)
            # print(output)

            loss = criterion(output, decision)
            test_loss += loss.item() * input.size(0)
            _, predicted = torch.max(output, 1)
            y_true.extend(decision.argmax(dim=1).numpy())
            y_pred.extend(predicted.numpy())

    # Calculate accuracy
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    test_accuracy = correct / len(y_true)
    test_loss /= len(test_dataset)  # Average test loss

    print(f"Model: {model_path}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    # Define activation functions
    activation_functions = {
        "identity": nn.Identity(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
        "sigmoid": nn.Sigmoid()
    }

    # Directory containing model weights
    model_dir = 'best_model_weights'
    model_files = os.listdir(model_dir)

    for model_file in model_files:
        if model_file.endswith('.pth'):
            model_path = os.path.join(model_dir, model_file)
            # Extract activation function and number of layers from the file name
            parts = model_file.split('_')
            activation_function_name = parts[0]
            num_hidden_layers = int(parts[1])
            activation_function = activation_functions.get(activation_function_name, nn.Identity())
            validate_model(model_path, activation_function, num_hidden_layers)
