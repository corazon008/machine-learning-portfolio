import torch

def compute_normalization(train_ds)-> tuple[list, list]:
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