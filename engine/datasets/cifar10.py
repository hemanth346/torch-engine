# from . import *
from engine.data_loaders import *
import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt

# logger = setup_logger(__name__)


# Constants
mean = (0.491, 0.482, 0.447)
std = (0.247, 0.243, 0.262)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transforms_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
]

# set batch size
batch_size: int = 64

# Other User inputs
seed = None
shuffle = True

# additional transformations
augmentations = []

# cuda attributes
num_workers = 0
pin_memory = True


# environment config
# CUDA?
cuda = torch.cuda.is_available()

if seed:
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


# dataloader cuda/cpu args
if cuda:
    dataloader_args = {
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'batch_size': batch_size }
else:
    dataloader_args = {
        'shuffle': shuffle,
        'batch_size': batch_size
    }


# print('Using cuda') if cuda else print('Using cpu')
# logger.info('Using cuda') if cuda else logger('Using cpu')


def data_loaders(transforms_list=transforms_list, augmentations=augmentations):
    """

    :param augmentations:
    :return:
    """
    if augmentations:
        transforms_list = [augmentations] + transforms_list

    train_transforms = transforms.Compose(transforms_list)
    test_transforms = transforms.Compose(transforms_list)

    # Get the Train and Test Set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

    dataloader_args['batch_size'] = batch_size

    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return train_loader, test_loader


def display_images(number=5, ncols=5, figsize=(10, 6), train=True):
    train_loader, test_loader = data_loaders(transforms_list, augmentations)
    loader = train_loader
    if not train:
        loader = test_loader
    show_images(loader, mean, std, classes, number, ncols, figsize)

