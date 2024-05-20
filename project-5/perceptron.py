import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.metrics import classification_report
from torchvision import datasets, transforms
from manage_data import BCDataset, prepare_dataset

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=30): 
        super(MLP, self).__init__() 
        self.num_hidden_layers = num_hidden_layers

        # input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # hidden layers with chosen activation function
        self.hidden_layers = self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num_hidden_layers)])

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
    
# Instantiate the custom module 
my_module = MLP(num_inputs=13, num_outputs=4, hidden_size=128) 

# Define the loss function and optimizer 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(my_module.parameters(), lr=0.01) 

# Define the transformations for the dataset 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) 

# Load the MNIST dataset 
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) 
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform) 

# Define the data loader 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True) 
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False) 

# Train the model 
for epoch in range(10): 
    for i, (images, labels) in enumerate(train_loader): 
        images = images.view(-1, 28*28) 
        optimizer.zero_grad() 
        output = my_module(images) 
        loss = criterion(output, labels) 
        loss.backward() 
        optimizer.step() 
    print('Epoch -->', epoch, '-->', loss.item()) 

# Test the model 
with torch.no_grad(): 
    y_true = [] 
    y_pred = [] 
    correct = 0
    total = 0
    for images, labels in test_loader: 
        images = images.view(-1, 28*28) 
        output = my_module(images) 
        _, predicted = torch.max(output.data, 1)  # Get the class with the highest score
        total += labels.size(0) 
        correct += (predicted == labels).sum().item()
        y_true += labels.tolist() 
        y_pred += predicted.tolist() 

    # Accuracy 
    print('Accuracy: {} %'.format(100 * correct / total)) 

    # Classification Report 
    report = classification_report(y_true, y_pred, digits=4) 
    print(report)