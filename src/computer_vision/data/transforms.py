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


def get_transform(img_size: int, mean: list, std: list, gray_scale: bool = False) -> transforms.Compose:
    t = transforms.Compose([
                               transforms.Resize((img_size, img_size)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                           ])
    if gray_scale:
        t.transforms.insert(len(t.transforms), transforms.Grayscale(num_output_channels=1))
    return t
