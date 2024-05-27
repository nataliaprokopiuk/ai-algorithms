import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from perceptron import MLP
from manage_data import prepare_dataset
import time
import os

if __name__ == "__main__":
    # Instantiate the custom module
    module = MLP(16, 32, 4)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(module.parameters(), lr=0.01, momentum=0.6)

    # Load datasets
    train_dataset, val_dataset, _ = prepare_dataset()

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Training parameters
    num_epochs = 10
    best_val_accuracy = 0.0

    best_model_state = None
    # Create the directory for saving model weights if it doesn't exist
    os.makedirs('best_model_weights', exist_ok=True)


    # Train the model
    for epoch in range(num_epochs):
        module.train()  # Set model to training mode
        running_loss = 0.0
        train_accuracy = 0.0
        train_loss = 0.0

        for input, decision in train_loader:
            # print()
            # print(input)
            # print(decision)
            optimizer.zero_grad()  # Zero the gradients
            # output = module.forward(input)  # Forward pass
            output = module(input)
            # print(output)
            # print()
            loss = criterion(output, decision)   # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item() * input.size(0)  # Accumulate loss

            _, predicted = torch.max(output, 1)  # Get predicted labels
            train_accuracy += (predicted == decision.argmax(dim=1)).sum().item()  # Compute accuracy

        train_loss = running_loss / len(train_dataset)  # Compute average loss
        train_accuracy /= len(train_dataset)  # Compute average accuracy

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        # Validation phase
        module.eval()  # Set model to evaluation mode
        val_accuracy = 0.0
        val_loss = 0.0

        with torch.no_grad():
            for input, decision in val_loader:
                # output = module.forward(input)  # Forward pass
                output = module(input)  # Forward pass
                loss = criterion(output, decision)  # Compute loss
                val_loss += loss.item() * input.size(0)  # Accumulate validation loss

                _, predicted = torch.max(output, 1)  # Get predicted labels
                # print()
                # print(decision)
                # print(predicted)
                # print()
                # print('decision ' + str(decision))
                # print((predicted == decision).sum().item())
                val_accuracy += (predicted == decision.argmax(dim=1)).sum().item()  # Compute accuracy

            val_loss /= len(val_dataset)  # Compute average validation loss
            val_accuracy /= len(val_dataset)  # Compute average accuracy

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save model weights if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            print("Best model updated")
            best_val_accuracy = val_accuracy
            best_model_state = module.state_dict()

        print("")

    # Save the best model weights at the end of the training process
    if best_model_state is not None:
        current_time = time.strftime('%Y-%m-%d %H-%M-%S')
        model_path = os.path.join('best_model_weights', f'best_model_weights_{current_time}.pth')
        torch.save(best_model_state, model_path)
        print(f"Best model weights saved to {model_path}")

    print("Training finished.")
