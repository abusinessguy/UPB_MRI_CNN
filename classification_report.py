import torch
from sklearn.metrics import classification_report
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

from ResNet18 import ResNetForClassification  # Your custom model import

def generate_classification_report(model, dataloader, device, class_names):
    """
    Generate a classification report for a trained model on a test dataset and save it as a table image.

    :param model: Trained PyTorch model.
    :param dataloader: DataLoader for the test dataset.
    :param device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').
    :param class_names: List of class names.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report as a dictionary
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Save classification report as an image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=report_df.round(2).values,
             colLabels=report_df.columns,
             rowLabels=report_df.index,
             cellLoc='center',
             loc='center')
    plt.title("Classification Report", fontsize=16)
    plt.savefig("classification_report.png", bbox_inches="tight")
    print("Classification report saved as 'classification_report.png'.")

if __name__ == "__main__":
    # Load your trained model
    model = ResNetForClassification(num_classes=4)  # Adjust num_classes as needed
    model.load_state_dict(torch.load("C:/Projects/MRI_CNN/trained_model_epoch_10.pth"))
    model.eval()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test dataset
    test_dir = "C:/Projects/MRI_CNN/dataset/test"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjusted to match training normalization
    ])
    test_dataset = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Class names
    class_names = test_dataset.classes
    print(test_dataset.classes)
 

    # Generate the classification report
    generate_classification_report(model, test_loader, device, class_names)
