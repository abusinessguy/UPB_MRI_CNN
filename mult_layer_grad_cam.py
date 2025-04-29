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


def visualize_multiple_layers(model, layers, input_tensor, class_idx=None, alpha=0.5):
    """
    Visualize Grad-CAM outputs for multiple layers side by side, both as overlays and standalone heatmaps.
    :param model: Trained model
    :param layers: List of layers to analyze
    :param input_tensor: Preprocessed input image tensor
    :param class_idx: Target class index for Grad-CAM
    :param alpha: Transparency for overlay visualization
    """
    # Denormalize the image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = input_tensor * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)
    image_pil = ToPILImage()(image_denorm.cpu())

    plt.figure(figsize=(6 * len(layers), 12))

    for i, layer in enumerate(layers):
        grad_cam = GradCAM(model, layer)
        cam = grad_cam(input_tensor, class_idx=class_idx)

        # Resize CAM to match image dimensions
        cam_resized = np.uint8(255 * cam)
        cam_resized = Image.fromarray(cam_resized).resize(image_pil.size, Image.BILINEAR)

        # Apply colormap to CAM
        cam_colored = plt.cm.viridis(np.array(cam_resized) / 255.0)[:, :, :3]  # Remove alpha channel
        cam_colored = (cam_colored * 255).astype(np.uint8)

        # Overlay CAM on original image
        cam_overlay = np.array(image_pil).astype(np.float32) * (1 - alpha) + cam_colored.astype(np.float32) * alpha
        cam_overlay = np.uint8(np.clip(cam_overlay, 0, 255))

        # Display Grad-CAM overlay
        plt.subplot(2, len(layers), i + 1)
        plt.imshow(cam_overlay)
        plt.title(f"Overlay Layer {i+1}")
        plt.axis('off')

        # Display Grad-CAM heatmap alone
        plt.subplot(2, len(layers), len(layers) + i + 1)
        plt.imshow(cam, cmap='viridis')
        plt.title(f"Grad-CAM Layer {i+1}")
        plt.axis('off')

    plt.show()


if __name__ == "__main__":
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    from PIL import Image

    # Load your trained model
    model = ResNetForClassification(num_classes=4)  # Adjust num_classes as needed
    model.load_state_dict(torch.load("C:/Projects/MRI_CNN/trained_model_epoch_10.pth"))
    model.eval()

    # Define layers to visualize
    layers_to_visualize = [
        model.base_model.layer1[-1].conv2,  # Layer 1
        model.base_model.layer2[-1].conv2,  # Layer 2
        model.base_model.layer3[-1].conv2,  # Layer 3
        model.base_model.layer4[-1].conv2   # Layer 4 (final conv layer)
    ]

    # Load and preprocess an example image
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image_path = r"C:\Projects\MRI_CNN\dataset\test\meningioma\Te-me_0010.jpg"  # Replace with the path to your image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)

    # Visualize Grad-CAM for multiple layers
    visualize_multiple_layers(model, layers_to_visualize, input_tensor)
