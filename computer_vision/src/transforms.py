from torchvision import transforms


def get_transform(img_size: int, mean: list, std: list, gray_scale: bool = False) -> transforms.Compose:
    t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    if gray_scale:
        t.transforms.insert(len(t.transforms), transforms.Grayscale(num_output_channels=1))
    return t
