import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import numpy as np


class CIFAR10Small:

    def __init__(self,
                 data_root="./data",
                 train_size=1000,
                 test_size=500,      
                 batch_size=64,
                 num_workers=0,
                 seed=0):

        self.data_root = data_root
        self.train_size = train_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self._build()

    def _build(self):
        # data augmentations for small data
        train_transform = T.Compose([
            #T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #T.Lambda(lambda x: x + 0.05 * torch.randn_like(x))
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            #T.Lambda(lambda x: x[[2,0,1], :, :]),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #T.Lambda(lambda x: x + 0.10 * torch.randn_like(x))
        ])

        full_train = torchvision.datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=True,
            transform=train_transform
        )

        full_test = torchvision.datasets.CIFAR10(
            root=self.data_root,
            train=False,
            download=True,
            transform=test_transform
        )

        # reproducible small subset for train
        np.random.seed(self.seed)
        train_indices = np.random.choice(len(full_train),
                                         self.train_size,
                                         replace=False)
        self.train_set = Subset(full_train, train_indices)

        # reproducible small subset for test, if test_size given
        if self.test_size is not None:
            test_indices = np.random.choice(len(full_test),
                                            self.test_size,
                                            replace=False)
            self.test_set = Subset(full_test, test_indices)
        else:
            self.test_set = full_test  # use full test set by default

    def loaders(self):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, test_loader


# Just to test the dataset class
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from collections import Counter

    data = CIFAR10Small(train_size=2000, test_size=500, batch_size=64)
    trainloader, testloader = data.loaders()

    print("Train size:", len(data.train_set))
    print("Test size:", len(data.test_set))

    # Class distribution
    labels = []
    for _, y in trainloader:
        labels.extend(y.tolist())

    counter = Counter(labels)
    print("\nClass distribution in training set:")
    for k in sorted(counter.keys()):
        print(f"Class {k}: {counter[k]} samples")

    # Visualize samples
    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    def unnormalize(img):
        return img * 0.5 + 0.5

    train_images, train_targets = next(iter(trainloader))
    test_images, test_targets = next(iter(testloader))

    fig, axes = plt.subplots(2, 6, figsize=(16, 6))

    for i in range(6):
        img = unnormalize(train_images[i]).permute(1, 2, 0)
        label = classes[train_targets[i].item()]
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Train: {label}", fontsize=10)
        axes[0, i].axis("off")

    for i in range(6):
        img = unnormalize(test_images[i]).permute(1, 2, 0)
        label = classes[test_targets[i].item()]
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Test: {label}", fontsize=10)
        axes[1, i].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Train vs Test Samples", fontsize=16)
    plt.subplots_adjust(hspace=0.2, wspace=0.3)
    plt.show()
