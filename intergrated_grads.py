import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from ResNet18 import ResNetForClassification

class IntegratedGradients:
    def __init__(self, model):
        self.model = model

    def compute_attributions(self, input_tensor, target_class, baseline=None, steps=50):
        """
        Compute Integrated Gradients attributions.

        :param input_tensor: Input image tensor (1 x C x H x W)
        :param target_class: Target class index for which attributions are computed.
        :param baseline: Baseline image tensor (same shape as input_tensor).
                         If None, uses a black image as the baseline.
        :param steps: Number of steps for the integration.
        :return: Attribution map as a tensor (C x H x W).
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Scale inputs and compute gradients
        scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)  # (steps + 1) x C x H x W

        self.model.eval()
        scaled_inputs.requires_grad = True

        # Forward pass
        outputs = self.model(scaled_inputs)
        target_outputs = outputs[:, target_class]

        # Backward pass to compute gradients
        grads = torch.autograd.grad(torch.sum(target_outputs), scaled_inputs)[0]

        # Average gradients and compute attributions
        avg_grads = grads[:-1].mean(dim=0)  # Exclude the final step
        attributions = (input_tensor - baseline).squeeze(0) * avg_grads

        return attributions


def visualize_multiple_scans(images, attributions):
    """
    Visualize Integrated Gradients attributions for multiple scans.

    :param images: List of original image tensors (C x H x W).
    :param attributions: List of attribution maps (C x H x W).
    :param titles: Classifier labels for each image.
    """
    num_images = len(images)
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        # Denormalize the input image for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_denorm = images[i] * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)

        image_pil = ToPILImage()(image_denorm.cpu())

        # Convert attributions to grayscale
        attribution = attributions[i].sum(dim=0).detach().cpu().numpy()
        attribution = np.clip(attribution, 0, None)  # Focus on positive attributions

        # Normalize attributions
        attribution -= attribution.min()
        attribution /= attribution.max()

        # Plot original image and attribution side by side
        plt.subplot(2, num_images+1, num_images + i)
        plt.imshow(image_pil)
        # plt.title(titles[i])
        plt.axis('off')

        plt.subplot(2, num_images, i + 1)
        plt.imshow(attribution, cmap="hot")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    from PIL import Image

    # Load your trained model
    model = ResNetForClassification(num_classes=4)  # Adjust num_classes as needed
    model.load_state_dict(torch.load("C:/Projects/MRI_CNN/trained_model_epoch_9.pth"))
    model.eval()

    # Load and preprocess multiple example images
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image_paths = [
        r"C:\Projects\MRI_CNN\misclassified_examples\20_4_true_meningioma_pred_no_tumor.png"
    ]

    # titles = ["Glioma", "No Tumor", "Pituitary", "Meningioma"]  # Classifier labels

    images = []
    attributions = []
    target_class = 0  # Replace with the desired class index

    ig = IntegratedGradients(model)

    for path in image_paths:
        image = Image.open(path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        attribution = ig.compute_attributions(input_tensor, target_class)

        images.append(input_tensor.squeeze(0))
        attributions.append(attribution)

    # Visualize attributions for multiple scans
    visualize_multiple_scans(images, attributions)
