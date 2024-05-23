import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from perceptron import MLP
from manage_data import prepare_dataset

if __name__ == "__main__":
    # Instantiate the custom module 
    module = MLP(16, 64, 4)

    # Define the loss function and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(module.parameters(), lr=0.01, momentum=0.9)

    # Load datasets
    train_dataset, val_dataset, _ = prepare_dataset()

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Training parameters
    num_epochs = 30
    best_val_accuracy = 0.0

    # Train the model 
    for epoch in range(num_epochs):
        module.train()  # Set model to training mode
        running_loss = 0.0

        for input, decision in train_loader:
            # print()
            # print(input)
            # print(decision)
            # print()
            optimizer.zero_grad()  # Zero the gradients
            # output = module.forward(input)  # Forward pass
            output = module(input)
            # loss = criterion(output, decision)  # Compute loss
            loss = criterion(output, decision)   # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item() * input.size(0)  # Accumulate loss

        epoch_loss = running_loss / len(train_dataset)  # Compute average loss

        # Validation phase
        module.eval()  # Set model to evaluation mode
        val_accuracy = 0.0

        with torch.no_grad():
            for input, decision in val_loader:
                # output = module.forward(input)  # Forward pass
                output = module(input)  # Forward pass
                _, predicted = torch.max(output, 1)  # Get predicted labels
                print(predicted)
                # print('decision ' + str(decision))
                # print((predicted == decision).sum().item())
                val_accuracy += (predicted == decision.argmax(dim=1)).sum().item()  # Compute accuracy

            val_accuracy /= len(val_dataset)  # Compute average accuracy

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save model weights if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            torch.save(module.state_dict(), 'best_model_weights.pth')
            best_val_accuracy = val_accuracy

    print("Training finished.")
