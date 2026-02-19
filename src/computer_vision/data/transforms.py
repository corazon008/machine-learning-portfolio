import torch
from torchvision import transforms


def compute_normalization(train_ds) -> tuple[list, list]:
    """Compute the mean and std of a dataset for normalization."""
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for img, _ in train_ds:
        for c in range(3):
            mean[c] += img[c, :, :].mean()
            std[c] += img[c, :, :].std()
    mean /= len(train_ds)
    std /= len(train_ds)
    return mean.tolist(), std.tolist()


def get_transform(img_size: int, mean: list, std: list, augments:bool = False, gray_scale: bool = False) -> transforms.Compose:
    t = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    if gray_scale:
        t.append(transforms.Grayscale(num_output_channels=1))
    if augments:
        t.append(transforms.RandomHorizontalFlip())
        t.append(transforms.RandomRotation(10))
    return transforms.Compose(t)

def inverse_transform(img: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """Inverse the normalization transform to get back the original image."""
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1 / s for s in std]
    inv_transform = transforms.Normalize(mean=inv_mean, std=inv_std)
    return inv_transform(img)