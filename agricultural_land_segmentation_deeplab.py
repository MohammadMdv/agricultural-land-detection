import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from agricultural_data_augmentation import prepare_dataloader


# Custom dataset class
class AgriculturalLandDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_mask.png'))

        # Debugging: Print paths to verify
        print(f"Image path: {img_path}")
        print(f"Mask path: {mask_path}")

        if os.path.isdir(img_path):
            raise RuntimeError(f"Image path is a directory: {img_path}")
        if os.path.isdir(mask_path):
            raise RuntimeError(f"Mask path is a directory: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        resize_transform = transforms.Resize((224, 224))
        image = resize_transform(image)
        mask = resize_transform(mask)

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']

        image = transforms.ToTensor()(image)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        # Ensure mask values are within the valid range
        mask = torch.clamp(mask, 0, 1)

        return image, mask


# Custom collate function
def custom_collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks


# Load pre-trained DeepLabV3+ model
def load_model(num_classes):
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model


# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']

            masks = masks.squeeze(1).long()
            print(f"Max value in masks: {masks.max()}")  # Debugging statement
            assert masks.max() < model.classifier[4].out_channels, "Target contains invalid class indices"

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


# Prediction function
def predict(model, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']

    pred = output.argmax(1).squeeze().cpu().numpy()
    return pred


# Visualization function
def visualize_prediction(image_path, pred):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(122)
    plt.imshow(pred, cmap='viridis')
    plt.title("Predicted Segmentation")
    plt.show()


# Main execution
def main():
    image_dir = "D:/cansat/cluade/image"
    mask_dir = "D:/cansat/cluade/mask"

    num_classes = 2  # Background and Agricultural Land
    dataset = AgriculturalLandDataset(image_dir, mask_dir)
    train_loader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn)

    model = load_model(num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    torch.save(model.state_dict(), "agricultural_land_model.pth")

    test_image_path = "D:/cansat/cluade/image/5.jpg"
    pred = predict(model, test_image_path)

    visualize_prediction(test_image_path, pred)


if __name__ == "__main__":
    main()
