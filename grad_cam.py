import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from ResNet18 import ResNetForClassification

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture gradients
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, class_idx):
        # Compute weights by global pooling of gradients
        weights = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)

        # Compute weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        # ReLU and normalize
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        return cam.cpu().detach().numpy()

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0)

        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[:, class_idx].backward()

        # Generate CAM
        return self.generate_cam(class_idx)


def visualize_cam(cam, image, alpha=0.5):
    """
    Display the original image, Grad-CAM heatmap, and an overlay of the two side by side.
    :param cam: The CAM heatmap (2D array)
    :param image: The original image (Tensor)
    :param alpha: Transparency factor for the overlay
    """
    from matplotlib.cm import get_cmap

    # Denormalize the image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = image * std + mean  # Reverse normalization
    image_denorm = torch.clamp(image_denorm, 0, 1)  # Clip values to valid range

    image_pil = ToPILImage()(image_denorm.cpu())

    # Resize CAM to match image dimensions
    cam_resized = np.uint8(255 * cam)
    cam_resized = Image.fromarray(cam_resized).resize(image_pil.size, Image.BILINEAR)
    cam_resized = np.array(cam_resized)

    # Apply colormap to the resized CAM
    colormap = get_cmap("viridis")
    cam_colored = colormap(cam_resized / 255.0)[:, :, :3]  # Apply colormap and remove alpha channel
    cam_colored = (cam_colored * 255).astype(np.uint8)  # Convert to 8-bit RGB

    # Overlay CAM on the original image
    cam_overlay = np.array(image_pil).astype(np.float32) * (1 - alpha) + cam_colored.astype(np.float32) * alpha
    cam_overlay = np.uint8(np.clip(cam_overlay, 0, 255))

    plt.figure(figsize=(18, 6))

    # Display original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_pil)
    plt.title("Original Image")
    plt.axis('off')

    # Display Grad-CAM heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='viridis')  # Default colormap for clarity
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    # Display overlay image
    plt.subplot(1, 3, 3)
    plt.imshow(cam_overlay)
    plt.title("Overlay Image")
    plt.axis('off')

    plt.show()



# Example usage
if __name__ == "__main__":
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    from PIL import Image

    # Load your trained model
    model = ResNetForClassification(num_classes=4)
    model.load_state_dict(torch.load("C:/Projects/MRI_CNN/trained_model_epoch_9.pth", weights_only=True))
    model.eval()


    # Set up Grad-CAM for the last convolutional layer
    target_layer = model.base_model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)

    # Load and preprocess an example image
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image_path = r"C:\Projects\MRI_CNN\misclassified_examples\20_4_true_meningioma_pred_no_tumor.png"  # Replace with the path to your image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    with torch.no_grad():
        input_tensor = transform(image).unsqueeze(0)
        predictions = torch.softmax(model(input_tensor), dim=1).squeeze()
        class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']#test_dataset.classes
        
        print("Prediction Probabilities:")
        for i, prob in enumerate(predictions):
            print(f"{class_labels[i]}: {prob.item():.4f}")
        
        predicted_class = torch.argmax(predictions).item()
        print(f"\nPredicted Class: {class_labels[predicted_class]}")
        input_tensor = transform(image).squeeze(0)

    # Generate and visualize Grad-CAM
    cam = grad_cam(input_tensor)
    visualize_cam(cam, input_tensor)
