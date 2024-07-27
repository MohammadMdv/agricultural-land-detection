import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# Define the input and output directories
input_dir = "./image"
input_mask_dir = "./mask"
output_dir = "./image_aug"
output_mask_dir = "./mask_aug"

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Define the augmentation pipeline
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.Transpose(p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
    ], p=0.8),
    A.CLAHE(p=0.8),
    A.RandomBrightnessContrast(p=0.8),    
    A.RandomGamma(p=0.8),
])

# Get list of input images
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Number of augmentations per image
num_aug_per_image = 10

# Augment each image and its mask
for image_file in tqdm(image_files, desc="Augmenting images"):
    # Read the image and mask
    image = cv2.imread(os.path.join(input_dir, image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask_file = os.path.splitext(image_file)[0] + "_mask.png"  # Assuming mask files have "_mask" suffix
    mask = cv2.imread(os.path.join(input_mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
    
    # Perform augmentation multiple times for each image
    for i in range(num_aug_per_image):
        # Apply the same augmentation to both image and mask
        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        
        # Save the augmented image and mask
        output_filename = f"{os.path.splitext(image_file)[0]}_aug_{i}.png"
        cv2.imwrite(os.path.join(output_dir, output_filename), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_mask_dir, output_filename), aug_mask)

print("Augmentation completed!")
