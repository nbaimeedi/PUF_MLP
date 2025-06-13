import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    data = df.values
    X = torch.tensor(data[:, :-1], dtype=torch.float32)
    y = torch.tensor(data[:, -1], dtype=torch.float32).view(-1, 1)
    return X, y

def train_model(X_train, y_train, X_test, y_test, epochs=50, lr=0.001):
    model = MLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_acc = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            preds = model(X_test)
            predicted = (preds >= 0.5).float()
            correct = (predicted == y_test).sum().item()
            accuracy = (correct / len(y_test)) * 100
            train_acc.append(accuracy)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

    # Plot accuracy vs epoch
    plt.plot(range(1, epochs + 1), train_acc, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.grid(True)
    plt.savefig("accuracy_plot.png")
    plt.show()

if __name__ == "__main__":
    # === Update these if needed ===
    train_file = "train_set.csv"
    test_file = "test_set_9.csv"

    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)

    train_model(X_train, y_train, X_test, y_test)
