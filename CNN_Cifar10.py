import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. Przygotowanie danych
transform = transforms.Compose(
    [transforms.ToTensor(),  # konwersja do tensoru
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # normalizacja
)

# Załadowanie zbioru treningowego i testowego CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 2. Zbudowanie sieci konwolucyjnej
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Warstwy konwolucyjne
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Warstwa w pełni połączona
        # Początkowo, musimy obliczyć rozmiar wyjścia z warstw konwolucyjnych
        self.fc1 = None  # Będzie inicjalizowane później

        self.fc2 = nn.Linear(512, 10)

        # Funkcja aktywacji ReLU oraz max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def get_conv_output_shape(self, x):
        # Przechodzimy przez wszystkie warstwy konwolucyjne i poolingowe
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.size(1), x.size(2), x.size(3)  # Zwracamy: (kanały, wysokość, szerokość)

    def forward(self, x):
        # Obliczamy rozmiar po przejściu przez konwolucje
        num_channels, height, width = self.get_conv_output_shape(x)

        # Inicjalizujemy fc1 z obliczonymi wymiarami
        if self.fc1 is None:
            self.fc1 = nn.Linear(num_channels * height * width, 512)

        x = self.pool(F.relu(self.conv1(x)))  # Konwolucja + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flattening
        x = F.relu(self.fc1(x))  # Warstwa w pełni połączona
        x = self.fc2(x)  # Wyjście
        return x

# Inicjalizacja modelu
model = CNN()

# 3. Funkcja straty i optymalizator
criterion = nn.CrossEntropyLoss()  # Strata dla klasyfikacji wieloklasowej
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optymalizator Adam

def train(model, trainloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()  # Zerowanie gradientów
            outputs = model(inputs)  # Propagacja w przód
            loss = criterion(outputs, labels)  # Obliczenie straty
            loss.backward()  # Propagacja wsteczna
            optimizer.step()  # Aktualizacja wag

            running_loss += loss.item()
            if i % 100 == 99:  # Co 100 batch, wypisz stratę
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

def test(model, testloader):
    model.eval()  # Ustawienie modelu w tryb testowy
    correct = 0
    total = 0
    with torch.no_grad():  # Brak potrzeby obliczania gradientów podczas testowania
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Uzyskanie predykcji
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy}%")

# Trening modelu
train(model, trainloader, criterion, optimizer, num_epochs=10)

# Testowanie modelu
test(model, testloader)