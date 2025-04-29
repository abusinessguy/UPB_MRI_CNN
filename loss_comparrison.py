import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
intermediary_training_results = pd.read_csv("intermediary_training_results.csv")
training_results = pd.read_csv("training_results.csv")

# Extract data for plotting
epochs_1 = intermediary_training_results['Epoch']
train_loss_1 = intermediary_training_results['Train Loss']
train_acc_1 = intermediary_training_results['Train Acc (%)']
test_loss_1 = intermediary_training_results['Test Loss']
test_acc_1 = intermediary_training_results['Test Acc (%)']

epochs_2 = training_results['Epoch']
train_loss_2 = training_results['Train Loss']
train_acc_2 = training_results['Train Acc (%)']
test_loss_2 = training_results['Test Loss']
test_acc_2 = training_results['Test Acc (%)']

# Plotting side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# Loss comparison
axes[0].plot(epochs_1, train_loss_1, label='Intermediary Train Loss', color='blue', linestyle='-')
axes[0].plot(epochs_1, test_loss_1, label='Intermediary Test Loss', color='blue', linestyle='--')
axes[0].plot(epochs_2, train_loss_2, label='Train Loss', color='orange', linestyle='-')
axes[0].plot(epochs_2, test_loss_2, label='Test Loss', color='orange', linestyle='--')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Comparison')
axes[0].legend()
axes[0].grid()

# Accuracy comparison
axes[1].plot(epochs_1, train_acc_1, label='Intermediary Train Accuracy', color='blue', linestyle='-')
axes[1].plot(epochs_1, test_acc_1, label='Intermediary Test Accuracy', color='blue', linestyle='--')
axes[1].plot(epochs_2, train_acc_2, label='Train Accuracy', color='orange', linestyle='-')
axes[1].plot(epochs_2, test_acc_2, label='Test Accuracy', color='orange', linestyle='--')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Comparison')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()
