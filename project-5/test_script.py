import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from manage_data import prepare_dataset
from torch.utils.data import DataLoader
from perceptron import MLP

if __name__ == "__main__":
    # Prepare test dataset
    _, _, test_dataset = prepare_dataset()

    # Data loader for test dataset
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model initialization
    module = MLP(input_size=13, hidden_size=64, output_size=4)

    # Load trained model weights
    module.load_state_dict(torch.load('best_model_weights.pth'))

    # Evaluation mode
    module.eval()

    # Lists to store true labels and predicted labels
    y_true = []
    y_pred = []

    # Iterate over test dataset and make predictions
    with torch.no_grad():
        for input, decision in test_loader:
            output = module(input)
            _, predicted = torch.max(output, 1)
            y_true.extend(decision.numpy())
            y_pred.extend(predicted.numpy())

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Test Accuracy:", accuracy)

    # Generate classification report
    report = classification_report(y_true, y_pred)
    print("\nClassification Report:\n", report)