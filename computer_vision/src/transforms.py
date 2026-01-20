# transforms.py
from torchvision import transforms


def get_transform(img_size: int, mean: list, std: list) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])