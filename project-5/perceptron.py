import torch
import torch.nn as nn
from sklearn.metrics import classification_report
import logging
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TensorBoard
writer = SummaryWriter()


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

    def train_model(self, model, train_loader, val_loader, criterion, optimizer, num_epochs):
        accuracy = Accuracy(task="multiclass", num_classes=4)
        best_model_weights = None
        best_val_accuracy = 0.0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            train_accuracy = 0.0

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.long().squeeze())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_accuracy += accuracy(outputs, targets.long().squeeze())

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = train_accuracy / len(train_loader)

            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.long().squeeze())
                    val_loss += loss.item()
                    val_accuracy += accuracy(outputs, targets.long().squeeze())

            val_epoch_loss = val_loss / len(val_loader)
            val_epoch_accuracy = val_accuracy / len(val_loader)

            logger.info(
                f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}')
            writer.add_scalar('Loss/val', val_epoch_loss, epoch)
            writer.add_scalar('Accuracy/val', val_epoch_accuracy, epoch)

            # Save the model weights if validation accuracy improves
            if val_epoch_accuracy > best_val_accuracy:
                best_val_accuracy = val_epoch_accuracy
                best_model_weights = self.state_dict()

        writer.close()

        # Save the best model weights to a file
        torch.save(best_model_weights, 'best_model_weights.pth')

# Test the model
    def test_model(model, test_loader):
        with torch.no_grad():
            y_true = []
            y_pred = []
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest score
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_true += labels.tolist()
                y_pred += predicted.tolist()

                # Accuracy
            accuracy = 100 * correct / total
            logger.info('Test Accuracy: {} %'.format(accuracy))

            # Classification Report
            report = classification_report(y_true, y_pred, digits=4)
            logger.info('\n' + report)