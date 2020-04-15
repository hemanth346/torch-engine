# from . import *
from engine.data_loaders import *
import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt



# logger = setup_logger(__name__)

seed = None
mean = (0.491, 0.482, 0.447)
std = (0.247, 0.243, 0.262)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# CUDA?
cuda = torch.cuda.is_available()

if seed:
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

# print('Using cuda') if cuda else print('Using cpu')
# logger.info('Using cuda') if cuda else logger('Using cpu')

# cuda attributes
shuffle = True
num_workers = 0
pin_memory = True

# dataloader cuda/cpu args
dataloader_args = {
    'shuffle': shuffle,
    'num_workers': num_workers,
    'pin_memory': pin_memory
} if cuda else {
    'shuffle': shuffle,
}

transforms_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
]


class transforms(Transforms):
    mean = mean
    std = std

    def __init__(self):
        super().__init__()

    def transforms_list(self, mean=mean, std=std):
        return super().transforms_list(mean, std)

    def train_transform(self, transforms_list=[], update=True):
        train_transforms = super().train_transform()
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
        train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
        return train_loader

    def test_transform(self, transforms_list=[], update=True):
        print(self.mean, self.std, mean, std)
        test_transforms = super().test_transform()
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
        test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)
        return test_loader


def get_loaders(batch_size:int=64, augmentations=[]):
    '''
    Load CIFAR10 data

    :params:
        batch_size : int
        augmentation transforms : List
        seed for dataloaders: int
    :returns:
        train dataloader
        test dataloader
    '''
    print(mean, std)
    transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]

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
    train_loader, test_loader = get_loaders()
    loader = train_loader
    if not train:
        loader = test_loader
    show_images(loader, mean, std, classes, number, ncols, figsize)

