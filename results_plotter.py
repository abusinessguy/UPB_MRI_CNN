import pandas as pd
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights

# Function to plot training results from a CSV file
def plot_from_csv(csv_file_path):
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Extract data for plotting
    epochs = data['Epoch']
    train_loss = data['Train Loss']
    test_loss = data['Test Loss']
    train_acc = data['Train Acc (%)']
    test_acc = data['Test Acc (%)']

    # Create a single figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Train and Test Loss
    axes[0].plot(epochs, train_loss, label="Train Loss", marker='o')
    axes[0].plot(epochs, test_loss, label="Test Loss", marker='o')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Test Loss")
    axes[0].legend()
    axes[0].grid()

    # Plot Train and Test Accuracy
    axes[1].plot(epochs, train_acc, label="Train Accuracy", marker='o')
    axes[1].plot(epochs, test_acc, label="Test Accuracy", marker='o')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Train vs Test Accuracy")
    axes[1].legend()
    axes[1].grid()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    csv_file_path = "training_results.csv"  # Path to the CSV file
    plot_from_csv(csv_file_path)
