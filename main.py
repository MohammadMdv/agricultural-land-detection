import cv2
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os


def load_model(model_path, num_classes=2):
    model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

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


def main():
    model_path = "agricultural_land_model.pth"
    input_dir = "D:/cansat/cluade/image"
    output_dir = "D:/cansat/cluade/output"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path)
    model.to(device)

    test_image_path = "D:/cansat/cluade/image/1.jpg"
    for i in range(1, 8):
        pred = predict(model, f"D:/cansat/cluade/image/{i}.jpg")
        cv2.imwrite(f"{i}_result.jpg", pred)

if __name__ == '__main__':
    main()