import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image
from matplotlib.colors import Normalize
from ResNet18 import ResNetForClassification
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def occlusion_sensitivity(model, input_tensor, target_class, occlusion_size=20, occlusion_stride=10, baseline_value=0):
    """
    Perform occlusion sensitivity analysis on a model and input image.
    """
    model.eval()
    _, _, height, width = input_tensor.shape
    sensitivity_map = np.zeros((height, width))

    with torch.no_grad():
        # Get the original prediction score for the target class
        original_output = model(input_tensor)
        original_score = original_output[0, target_class].item()

        # Slide the occlusion patch over the image
        for y in range(0, height - occlusion_size + 1, occlusion_stride):
            for x in range(0, width - occlusion_size + 1, occlusion_stride):
                # Create a copy of the input tensor
                occluded_tensor = input_tensor.clone()

                # Apply the occlusion patch
                occluded_tensor[:, :, y:y + occlusion_size, x:x + occlusion_size] = baseline_value

                # Get the prediction score for the occluded image
                occluded_output = model(occluded_tensor)
                occluded_score = occluded_output[0, target_class].item()

                # Calculate the sensitivity (drop in score)
                sensitivity_map[y:y + occlusion_size, x:x + occlusion_size] += original_score - occluded_score

    # Debug: Print sensitivity map stats before normalization
    print(f"Sensitivity map (raw) stats: min={sensitivity_map.min()}, max={sensitivity_map.max()}")

    # Normalize the sensitivity map
    sensitivity_map -= sensitivity_map.min()
    sensitivity_map /= sensitivity_map.max()

    # Debug: Print sensitivity map stats after normalization
    print(f"Sensitivity map (normalized) stats: min={sensitivity_map.min()}, max={sensitivity_map.max()}")

    return sensitivity_map

def visualize_occlusion_sensitivity(model, input_tensor, sensitivity_map, alpha=0.5):
    """
    Visualize the occlusion sensitivity heatmap with an overlay on the original image.
    """
    # Denormalize the image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = input_tensor * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)

    image_pil = ToPILImage()(image_denorm.squeeze(0).cpu())

    # Resize sensitivity map
    print(f"Resizing sensitivity map from shape {sensitivity_map.shape} to image size {image_pil.size}")
    sensitivity_resized = np.array(Image.fromarray(sensitivity_map).resize(image_pil.size, Image.BILINEAR))

    # Debug: Print resized sensitivity map stats
    print(f"Resized sensitivity map stats: min={sensitivity_resized.min()}, max={sensitivity_resized.max()}")

    # Normalize resized sensitivity map for visualization
    sensitivity_resized = (sensitivity_resized - sensitivity_resized.min()) / (sensitivity_resized.max() - sensitivity_resized.min())

    # Apply colormap
    sensitivity_colored = plt.cm.hot(sensitivity_resized)[:, :, :3]

    # Create overlay with blending
    overlay = np.array(image_pil).astype(np.float32) * (1 - alpha) + (sensitivity_colored * 255).astype(np.float32) * alpha
    overlay = np.uint8(np.clip(overlay, 0, 255))


    # Plot results
    fig, axs = plt.subplots(1, 4, figsize=(20, 6), gridspec_kw={"width_ratios": [1, 1, 1, 0.1]})

    # Original image
    axs[0].imshow(image_pil)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Sensitivity heatmap
    norm = Normalize(vmin=0, vmax=1)
    heatmap = axs[1].imshow(sensitivity_resized, cmap='hot', norm=norm)
    axs[1].set_title("Occlusion Heatmap")
    axs[1].axis('off')

    # Overlay image
    axs[2].imshow(overlay)
    axs[2].set_title("Overlay Image")
    axs[2].axis('off')

    # Add colorbar to the far right
    cbar = fig.colorbar(heatmap, cax=axs[3], fraction=0.046, pad=0.04)
    cbar.set_label('Sensitivity', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import torchvision.transforms as transforms

    test_dir = "C:/Projects/MRI_CNN/dataset/test"
    test_dataset = ImageFolder(test_dir)

    # Load your trained model
    model = ResNetForClassification(num_classes=4)
    model.load_state_dict(torch.load("C:/Projects/MRI_CNN/trained_model_epoch_9.pth", weights_only=True))
    model.eval()

    # Load and preprocess an example image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image_path = r"C:\Projects\MRI_CNN\misclassified_examples\20_4_true_meningioma_pred_no_tumor.png"
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0)

    # Automatically determine the target class
    with torch.no_grad():
        predictions = torch.softmax(model(input_tensor), dim=1).squeeze()
        target_class = torch.argmax(predictions).item()

    print(f"Target class automatically determined: {target_class}")

    # Perform occlusion sensitivity analysis
    sensitivity_map = occlusion_sensitivity(model, input_tensor, target_class)

    # Visualize the results
    visualize_occlusion_sensitivity(model, input_tensor, sensitivity_map)
