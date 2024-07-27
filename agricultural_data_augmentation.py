import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


class AugmentedAgriculturalDataset(Dataset):
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

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure image is in the correct shape (C, H, W)
        if image.shape[0] != 3:
            image = image.permute(2, 0, 1)

        # Convert mask to tensor manually and ensure it's 2D
        mask = torch.tensor(mask, dtype=torch.long).squeeze()

        # Ensure mask values are within the valid range
        mask = torch.clamp(mask, 0, 1)

        return image, mask


# Define augmentation pipeline
def get_train_augmentation():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Resize(height=224, width=224, always_apply=True),  # Ensure consistent size
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        ], p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def visualize_augmentations(dataset, num_samples=5):
    fig, axs = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    for i in range(num_samples):
        image, mask = dataset[i % len(dataset)]
        axs[i, 0].imshow(image.permute(1, 2, 0))
        axs[i, 0].set_title('Augmented Image')
        axs[i, 0].axis('off')
        axs[i, 1].imshow(mask, cmap='gray')
        axs[i, 1].set_title('Augmented Mask')
        axs[i, 1].axis('off')
    plt.tight_layout()
    plt.show()


def collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images)
    masks = torch.stack(masks)
    return images, masks

def prepare_dataloader(image_dir, mask_dir, batch_size=4):
    train_dataset = AugmentedAgriculturalDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=get_train_augmentation()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    return train_loader

if __name__ == "__main__":
    # Set paths to your image and mask directories
    image_dir = "./image"
    mask_dir = "./mask"

    # Create the augmented dataset
    augmented_dataset = AugmentedAgriculturalDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=get_train_augmentation()
    )

    # Visualize some augmented samples
    visualize_augmentations(augmented_dataset, num_samples=5)

    # Prepare the DataLoader
    train_loader = prepare_dataloader(image_dir, mask_dir)

    print(f"Number of batches in train_loader: {len(train_loader)}")
