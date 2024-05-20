import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from perceptron import MLP
from manage_data import prepare_dataset

if __name__ == "__main__":
    # Prepare datasets
    train_dataset, val_dataset, _ = prepare_dataset()

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model initialization
    model = MLP(input_size=13, hidden_size=64, output_size=4)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training parameters
    num_epochs = 10
    best_val_accuracy = 0.0

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate over training dataset
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets.long().squeeze())  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item() * inputs.size(0)  # Accumulate loss

        epoch_loss = running_loss / len(train_dataset)  # Compute average loss

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_accuracy = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get predicted labels
                val_accuracy += (predicted == targets).sum().item()  # Compute accuracy

            val_accuracy /= len(val_dataset)  # Compute average accuracy

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save model weights if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            torch.save(model.state_dict(), 'best_model_weights.pth')
            best_val_accuracy = val_accuracy

    print("Training finished.")
