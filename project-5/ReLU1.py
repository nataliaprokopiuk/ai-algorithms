# Następnie dla 1 warstwy ukrytej oraz funkcji aktywacji ReLU proszę zbadać wpływ liczby
# neuronów w warstwie ukrytej na wyniki. Czy pojawiło się niedouczenie lub przeuczenie?
# Dlaczego? Jak temu zaradzić? Proszę zastosować środki zapobiegawcze przeuczeniu, takie
# jak odrzucanie oraz regularyzacja L2. Jak wpłynęły one na wyniki?

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from perceptron import MLP
from manage_data import prepare_dataset
import time
import os
import pandas as pd

def train_model_and_get_results(num_hidden_neurons):
    # Instantiate the custom module
    module = MLP(16, num_hidden_neurons, 4, activation_function=nn.ReLU(), num_hidden_layers=1)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(module.parameters(), lr=0.01, momentum=0.6)

    # Load datasets
    train_dataset, val_dataset, _ = prepare_dataset()

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Training parameters
    num_epochs = 50

    # Lists to store training and validation results
    train_results = []
    val_results = []

    # Train the model
    for epoch in range(num_epochs):
        module.train()  # Set model to training mode
        running_loss = 0.0
        train_accuracy = 0.0

        for input, decision in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            output = module(input)
            loss = criterion(output, decision)   # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item() * input.size(0)  # Accumulate loss

            _, predicted = torch.max(output, 1)  # Get predicted labels
            train_accuracy += (predicted == decision.argmax(dim=1)).sum().item()  # Compute accuracy

        train_loss = running_loss / len(train_dataset)  # Compute average loss
        train_accuracy /= len(train_dataset)  # Compute average accuracy

        # Validation phase
        module.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for input, decision in val_loader:
                output = module(input)  # Forward pass
                loss = criterion(output, decision)  # Compute loss
                val_loss += loss.item() * input.size(0)  # Accumulate validation loss

                _, predicted = torch.max(output, 1)  # Get predicted labels
                val_accuracy += (predicted == decision.argmax(dim=1)).sum().item()  # Compute accuracy

            val_loss /= len(val_dataset)  # Compute average validation loss
            val_accuracy /= len(val_dataset)  # Compute average validation accuracy

        # Append results for this epoch
        train_results.append({'Epoch': epoch + 1, 'Loss': train_loss, 'Accuracy': train_accuracy})
        val_results.append({'Epoch': epoch + 1, 'Loss': val_loss, 'Accuracy': val_accuracy})

    return pd.DataFrame(train_results), pd.DataFrame(val_results)

if __name__ == "__main__":
    num_hidden_neurons_list = [10, 50, 100, 200]  # List of different numbers of neurons in the hidden layer

    # Dictionary to store results for different number of neurons
    results_dict = {}

    for num_hidden_neurons in num_hidden_neurons_list:
        train_results, val_results = train_model_and_get_results(num_hidden_neurons)
        results_dict[num_hidden_neurons] = {'Train Results': train_results, 'Validation Results': val_results}

    # Save results to CSV files
    for num_hidden_neurons, results in results_dict.items():
        train_results_file = f'train_results_{num_hidden_neurons}_neurons.csv'
        val_results_file = f'val_results_{num_hidden_neurons}_neurons.csv'

        results['Train Results'].to_csv(train_results_file, index=False)
        results['Validation Results'].to_csv(val_results_file, index=False)
