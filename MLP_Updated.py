import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load dataset from CSV
def load_dataset(file_path):
    df = pd.read_csv(file_path, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)
    return torch.tensor(X), torch.tensor(y)

# Train and evaluate model
def train_model(X_train, y_train, X_test, y_test, epochs=200, lr=0.01):
    model = MLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test)
            predicted = (preds >= 0.5).float()
            correct = (predicted == y_test).sum().item()
            accuracy = (correct / len(y_test)) * 100
            test_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%")

    # Save final predictions
    with torch.no_grad():
        final_preds = model(X_test)
        final_labels = (final_preds >= 0.5).float()

        expected_np = y_test.squeeze().numpy().astype(int)
        predicted_np = final_labels.squeeze().numpy().astype(int)
        matched_text = np.where(expected_np == predicted_np, "Matched", "Not Matched")

        df_out = pd.DataFrame({
            'Expected': expected_np,
            'Predicted': predicted_np,
            'Match Status': matched_text
        })

        # Save predictions
        df_out.to_csv("predictions_train_set.csv", index=False)

        # Append final accuracy at the bottom
        with open("predictions_train_set.csv", "a") as f:
            f.write(f"\n,,Final Accuracy: {test_accuracies[-1]:.2f}%\n")

        print("\nSaved predictions to predictions_train_set.csv")
        print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")

    # Plot accuracy vs epoch
    plt.plot(range(1, epochs + 1), test_accuracies, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Epoch")
    plt.grid(True)
    plt.savefig("train_set_plot.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    X_train, y_train = load_dataset("train_set.csv")
    X_test, y_test = load_dataset("train_set.csv")  # Change to test_set_2.csv etc. if needed

    # Train model
    train_model(X_train, y_train, X_test, y_test, epochs=200, lr=0.01)
