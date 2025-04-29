import os
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from ResNet18 import ResNetForClassification  # Your custom model import
import torch.nn.functional as F  # For softmax


def save_misclassified_examples(model, dataloader, device, class_names, output_dir="misclassified_examples"):
    """
    Identify and save misclassified examples from the test dataset.
    :param model: Trained PyTorch model.
    :param dataloader: DataLoader for the test dataset.
    :param device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').
    :param class_names: List of class names.
    :param output_dir: Directory to save misclassified examples.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)  # Compute softmax probabilities
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size(0)):
                if preds[j] != labels[j]:
                    # Convert tensor to image
                    img = inputs[j].cpu().permute(1, 2, 0).numpy()
                    img = (img * 0.5 + 0.5)  # Undo normalization (mean=0.5, std=0.5)
                    img = (img * 255).astype("uint8")
                    
                    # Get true label, predicted label, and softmax probabilities
                    true_label = class_names[labels[j].item()]
                    pred_label = class_names[preds[j].item()]
                    softmax_output = probs[j].cpu().numpy()
                    
                    # Save the misclassified image
                    file_name = f"{i}_{j}_true_{true_label}_pred_{pred_label}.png"
                    file_path = os.path.join(output_dir, file_name)
                    plt.imsave(file_path, img)
                    
                    # Print details of the misclassified image
                    print(f"Misclassified Image: {file_name}")
                    print(f"True Label: {true_label}")
                    print(f"Predicted Label: {pred_label}")
                    print(f"Softmax Outputs: {softmax_output}")
                    print("-" * 50)

    print(f"Misclassified examples saved in '{output_dir}'.")


if __name__ == "__main__":
    # Load your trained model
    model = ResNetForClassification(num_classes=4)  # Adjust num_classes as needed
    model.load_state_dict(torch.load("C:/Projects/MRI_CNN/trained_model.pth"))
    model.eval()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test dataset
    test_dir = "C:/Projects/MRI_CNN/dataset/test"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjusted to match training normalization
    ])
    test_dataset = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Class names
    class_names = test_dataset.classes

    # Save misclassified examples
    save_misclassified_examples(model, test_loader, device, class_names)
