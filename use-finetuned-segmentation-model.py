import os

# Set the environment variable at the very beginning of the script
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")


def get_model(num_classes, aux_loss=True):
    model = deeplabv3_resnet101(pretrained=False, aux_loss=aux_loss)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    if aux_loss:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model


def load_model(model, checkpoint_path, num_classes):
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()

    # Filter out mismatched keys
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Reinitialize the classifier layers
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    return model


def segment_image(model, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']

    return output.argmax(1).squeeze().cpu().numpy()


def visualize_segmentation(image_path, segmentation):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmentation, cmap='jet')
    plt.title("Segmentation Result")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set the number of classes for your current task
    num_classes = 2  # Adjust this if you have a different number of classes

    # Load the model
    model = get_model(num_classes, aux_loss=True)
    model = load_model(model, "fine_tuned_segmentation_model_resenet101.pth", num_classes)

    # Path to your test image
    test_image_path = "./image/6.jpg"

    # Perform segmentation
    segmentation_result = segment_image(model, test_image_path)

    # Visualize the result
    visualize_segmentation(test_image_path, segmentation_result)