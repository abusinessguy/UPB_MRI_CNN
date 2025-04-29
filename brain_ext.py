import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random


def edge_detection(input_path, lower_threshold=50, upper_threshold=150):
    """
    Apply Canny edge detection to isolate the brain region.

    :param input_path: Path to the input image file.
    :param lower_threshold: Lower bound for Canny edge detection.
    :param upper_threshold: Upper bound for Canny edge detection.
    :return: Edge-detected image as a binary mask.
    """
    # Load the image and convert to grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = cv2.Canny(img, lower_threshold, upper_threshold)

    # Fill the inside of the edges
    filled_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel=np.ones((15, 15), np.uint8))

    # Find the largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_edges, connectivity=8)

    # Get the label of the largest connected component (excluding the background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip label 0 (background)

    # Create a binary mask for the largest component
    largest_component = (labels == largest_label).astype(np.uint8) * 255

    # Fill the inside of the largest component completely
    kernel = np.ones((25, 25), np.uint8)  # Larger kernel to ensure full filling
    filled_component = cv2.morphologyEx(largest_component, cv2.MORPH_CLOSE, kernel)

    return filled_component

def flood_fill_background(edges):
    """
    Use flood fill to isolate the background and create a mask for the brain region.

    :param edges: Edge-detected binary image.
    :return: Binary mask isolating the brain region.
    """
    h, w = edges.shape
    flood_filled = edges.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, mask, (0, 0), 255)

    # Invert flood-filled image to get the brain region mask
    brain_mask = cv2.bitwise_not(flood_filled)

    # Combine the original edges with the brain mask to preserve borders
    brain_mask_with_edges = cv2.bitwise_or(brain_mask, edges)

    return brain_mask_with_edges

def visualize_processing_steps(original_path, lower_threshold=50, upper_threshold=150):
    """
    Visualize the original image and the final flood-filled mask.

    :param original_path: Path to the original image file.
    :param lower_threshold: Lower bound for Canny edge detection.
    :param upper_threshold: Upper bound for Canny edge detection.
    """
    # Perform edge detection
    edges = edge_detection(original_path, lower_threshold, upper_threshold)

    # Perform flood fill to isolate the brain region
    brain_mask = flood_fill_background(edges)

    # Load the original image for visualization
    original = Image.open(original_path).convert("L")

    # Plot the results
    plt.figure(figsize=(10, 5))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    # Plot flood-filled mask
    plt.subplot(1, 2, 2)
    plt.imshow(brain_mask, cmap="gray")
    plt.title("Flood-Filled Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define the root directory containing MRI images
    root_dir = "C:/Projects/MRI_CNN/dataset/test"

    # Collect all image file paths
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(subdir, file))

    # Randomly select an image to visualize
    random_image_path = random.choice(image_paths)

    # Visualize the processing steps for the selected image
    visualize_processing_steps(random_image_path, lower_threshold=50, upper_threshold=150)
