import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def fashionmnist(batch_size=5, size=28):
    labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )

    train_loader = DataLoader(
        dataset=datasets.FashionMNIST(
            root="data", train=True, transform=transform, download=True
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=datasets.FashionMNIST(
            root="data", train=False, transform=transform, download=True
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, test_loader, labels


def cifar10(batch_size=5, size=32):
    labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )

    train_loader = DataLoader(
        dataset=datasets.CIFAR10(
            root="data", train=True, transform=transform, download=True
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=datasets.CIFAR10(
            root="data", train=False, transform=transform, download=True
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, test_loader, labels


def tiny_imagenet(batch_size=5, size=64):
    num_class = 200
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )
    train_loader = DataLoader(
        dataset=datasets.ImageFolder(
            root="data/tiny-imagenet-200/train", transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=datasets.ImageFolder(
            root="data/tiny-imagenet-200/val", transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, test_loader, num_class


def my_imagenet(batch_size=5, size=224):
    labels = [
        "tench",
        "brambling",
        "goldfinch",
        "house finch",
        "snowbird",
        "indigo bunting",
        "robin",
        "bulbul",
        "jay",
        "magpie",
    ]
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )

    train_loader = DataLoader(
        dataset=datasets.ImageFolder(root="data/Imagenet/train", transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=datasets.ImageFolder(root="data/Imagenet/val", transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, test_loader, labels
