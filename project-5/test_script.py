import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from manage_data import prepare_dataset
from torch.utils.data import DataLoader
from perceptron import MLP
import os

if __name__ == "__main__":
    # Prepare test dataset
    _, _, test_dataset = prepare_dataset()

    # Data loader for test dataset
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Model initialization
    module = MLP(input_size=16, hidden_size=32, output_size=4)

    # 1
    # Get the newest model file in the directory
    # model_dir = 'best_model_weights'
    # model_files = os.listdir(model_dir)
    # if not model_files:
    #     print("No model files found in the directory:", model_dir)
    #     exit()
    # newest_model_file = max(model_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
    # # Load trained model weights
    # module.load_state_dict(torch.load(os.path.join('best_model_weights', newest_model_file)))

    # 2
    # Enter the file name
    model_file_name = "best_model_weights_2024-05-27 21-08-21.pth"
    # Check if the specified file exists
    model_dir = 'best_model_weights'
    model_path = os.path.join(model_dir, model_file_name)
    if not os.path.isfile(model_path):
        print(f"File '{model_file_name}' does not exist in the directory:", model_dir)
        exit()

    # Load trained model weights
    module.load_state_dict(torch.load(model_path))

    # Evaluation mode
    module.eval()

    # Lists to store true labels and predicted labels
    y_true = []
    y_pred = []

    # Iterate over test dataset and make predictions
    with torch.no_grad():
        for input, decision in test_loader:
            output = module(input)
            print(output)
            _, predicted = torch.max(output, 1)
            y_true.extend(decision.argmax(dim=1).numpy())
            y_pred.extend(predicted.numpy())

    # Calculate accuracy
    # print(y_true)
    # print(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print("Test Accuracy:", accuracy)

    # Generate classification report
    report = classification_report(y_true, y_pred)
    print("\nClassification Report:\n", report)