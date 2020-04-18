import engine.utils.plots
from engine import datasets
from engine import utils
import torch
import torchvision
import torchvision.transforms as transforms

# logger = setup_logger(__name__)
__all__ = ['mean', 'std', 'classes', 'transforms_list', 'dataloader_args', 'data_loaders', 'show_images']


# Constants
mean = (0.491, 0.482, 0.447)
std = (0.247, 0.243, 0.262)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transforms_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
]

# default dataloader_args for all datasets, available to modify as attribute
dataloader_args: dict = datasets.dataloader_args


def data_loaders(batch_size=64, augmentations=[]):
    """

    :param batch_size:
    :param augmentations:
    :return: Data loaders
    """
    train_transform_list = transforms_list

    if augmentations:
        train_transform_list = [augmentations] + transforms_list

    train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose(transforms_list)

    # Get the Train and Test Set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

    dataloader_args['batch_size'] = batch_size

    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return train_loader, test_loader


def show_images(data_loader, number=5, ncols=5, figsize=(10, 6)):
    engine.utils.plots.show_images(data_loader, mean, std, classes, number, ncols, figsize)
