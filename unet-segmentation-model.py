import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# U-Net model definition
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


# Custom dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(224, 224)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.images = [img for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize both image and mask to target size
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        if self.transform:
            image = self.transform(image)

        # Convert mask to tensor
        mask = transforms.ToTensor()(mask)

        return image, mask.squeeze().long()


# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")


# Inference function
def segment_image(model, image_path, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    return output.argmax(1).squeeze().cpu().numpy()


# Visualization function
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
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up paths
    train_image_dir = "./image_aug"
    train_mask_dir = "./mask_aug"

    # Set up transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and dataloader
    target_size = (224, 224)  # You can adjust this size based on your needs
    train_dataset = SegmentationDataset(train_image_dir, train_mask_dir, transform=transform, target_size=target_size)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Initialize the model
    model = UNet(n_channels=3, n_classes=2).to(device)

    # Set up loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 50
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # Save the trained model
    torch.save(model.state_dict(), "unet_segmentation_model.pth")
