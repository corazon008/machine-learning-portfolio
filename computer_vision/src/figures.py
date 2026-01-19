import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_label_distribution(ds, figures_name):

    if figures_name.exists():
        img = plt.imread(figures_name)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    else:
        label_counts = dict()
        for _, label in ds:
            label_name = ds.classes[label]
            if label_name in label_counts:
                label_counts[label_name] += 1
            else:
                label_counts[label_name] = 1

        sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
        plt.xticks(rotation=45)
        plt.title("Distribution of classes in the training set")
        plt.savefig(figures_name)
        plt.show()


def plot_example(X, y, classes, n=5, mean=None, std=None):
    """Plot the images in X and their labels in rows of `n` elements."""
    fig = plt.figure(figsize=(10, 10))
    rows = len(X) // n + 1
    for i, (img, y) in enumerate(zip(X, y)):
        ax = fig.add_subplot(rows, n, i + 1)
        if mean is not None and std is not None:
            img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(classes[y])
    plt.tight_layout()
    return fig