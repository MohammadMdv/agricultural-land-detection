import os

# Set the environment variable at the very beginning of the script
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.dc1 = double_conv(n_channels, 64)
        self.dc2 = double_conv(64, 128)
        self.dc3 = double_conv(128, 256)
        self.dc4 = double_conv(256, 512)
        self.dc5 = double_conv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dc6 = double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dc7 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dc8 = double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dc9 = double_conv(128, 64)

        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.dc1(x)
        x2 = self.dc2(nn.MaxPool2d(2)(x1))
        x3 = self.dc3(nn.MaxPool2d(2)(x2))
        x4 = self.dc4(nn.MaxPool2d(2)(x3))
        x5 = self.dc5(nn.MaxPool2d(2)(x4))

        x = self.up1(x5)
        x = self.dc6(torch.cat([x4, x], dim=1))
        x = self.up2(x)
        x = self.dc7(torch.cat([x3, x], dim=1))
        x = self.up3(x)
        x = self.dc8(torch.cat([x2, x], dim=1))
        x = self.up4(x)
        x = self.dc9(torch.cat([x1, x], dim=1))

        return self.final(x)


def load_model(model_path, device):
    model = UNet(n_channels=3, n_classes=2)  # Adjusted to match the trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the size expected by the model
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Normalize based on ImageNet statistics
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)


def visualize_segmentation(image_path, output):
    image = Image.open(image_path).convert("RGB")
    output = output.squeeze().cpu().numpy()
    output = np.argmax(output, axis=0)  # Convert the 2-channel output to a single-channel mask
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(output, cmap="gray")
    plt.show()


# function to visualize multiple images and their segmented counter-parts
def visualize_multiple_segmentations(image_paths, segmentation_outputs):
    """
    Visualizes multiple images and their segmented counterparts.

    Parameters:
    - image_paths: List of paths to the original images.
    - segmentation_outputs: List of numpy arrays representing the segmented images.
    """
    num_images = len(image_paths)
    plt.figure(figsize=(10, num_images * 2))

    for i, (image_path, segmentation) in enumerate(zip(image_paths, segmentation_outputs), start=1):
        # Load and display the original image
        image = Image.open(image_path).convert("RGB")
        plt.subplot(num_images, 2, 2 * i - 1)
        plt.imshow(image)
        plt.title(f"Original Image {i}")
        plt.axis('off')

        # Display the segmented image
        plt.subplot(num_images, 2, 2 * i)
        output = segmentation.squeeze().cpu().numpy()
        output = np.argmax(output, axis=0)  # Convert the 2-channel output to a single-channel mask
        plt.imshow(output, cmap='gray')  # Assuming segmentation is a single-channel image
        plt.title(f"Segmented Image {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def main(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    image = preprocess_image(image_path, device)
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)  # Assuming the model uses a sigmoid activation for the final layer
    visualize_segmentation(image_path, output)


def main_multiple(image_paths, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    images = [preprocess_image(image_path, device) for image_path in image_paths]
    with torch.no_grad():
        outputs = [torch.sigmoid(model(image)) for image in images]
    visualize_multiple_segmentations(image_paths, outputs)


# Example usage:
# main("./image/1.jpg", "unet_segmentation_model.pth")
main_multiple([f"./image/{i}.jpg" for i in range(1, 8)], "unet_segmentation_model.pth")
