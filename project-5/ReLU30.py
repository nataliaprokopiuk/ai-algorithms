# . Dla liczby warstw ukrytych 30 i ReLU proszę
# dla macierzy wag każdej warstwy zaraportować średnią normę (ang. matrix norm)
# gradientów w czasie pierwszej epoki trenowania

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
        loss.backward()  # Wsteczny przebieg
        optimizer.step()  # Aktualizuj wagi
        running_loss += loss.item() * input.size(0)  # Skumuluj stratę

    return running_loss / len(train_loader.dataset)  # Zwróć średnią stratę


if __name__ == "__main__":
    # Zdefiniuj funkcję aktywacji i liczbę warstw ukrytych
    activation_function = nn.ReLU()
    num_hidden_layers = 30

    # Przygotuj zbiór danych treningowych
    train_dataset, _, _ = prepare_dataset()
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Inicjalizuj model
    model = MLP(16, 32, 4, activation_function=activation_function, num_hidden_layers=num_hidden_layers)

    # Definiuj funkcję straty i optymalizator
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6)

    # Trenuj model przez jedną epokę, aby uzyskać gradienty
    train_loss = train_model(model, train_loader, criterion, optimizer)

    # Oblicz normy gradientów dla każdej warstwy w macierzy wag
    weight_gradients = []
    for param in model.parameters():
        if param.grad is not None:
            weight_gradients.append(param.grad.norm().item())

    # Oblicz średnią normę gradientów
    average_gradient_norm = sum(weight_gradients) / len(weight_gradients)

    print(f"Średnia norma gradientów w czasie pierwszej epoki trenowania: {average_gradient_norm}")
