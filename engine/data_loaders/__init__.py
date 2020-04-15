import abc
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from engine.utils import *

__all__ = ['Transforms', 'make_grid', 'set_plt_param', 'show_images']


class Transforms(abc.ABC):
    # @abc.abstractmethod
    # def __init__(self):
    #     self.mean = None
    #     self.std = None
    #
    # @property
    # @classmethod
    # @abc.abstractmethod
    # def mean(self):
    #     """
    #         Attribute to be defined in all child classes
    #     :return:
    #     """
    #     return self.mean
    #
    # @mean.setter
    # def mean(self, value):
    #     self.mean = value
    #
    # @property
    # @classmethod
    # @abc.abstractmethod
    # def std(self):
    #     """
    #         Attribute to be defined in all child classes
    #     :return:
    #     """
    #     return self.std
    #
    # @std.setter
    # def std(self, value):
    #     self.std = value
    @abc.abstractmethod
    def transforms_list(self, mean, std):
        return [
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ]
    # transforms_list = [
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std)
    #         ]

    @abc.abstractmethod
    def train_transform(self, transforms_list=[], update=True):
        # if no input transforms, use existing
        tlist = self.transforms_list
        if transforms_list:
            if update:
                # append input transforms to existing list
                tlist = self.transforms_list + transforms_list
            else:
                # use input transforms as is without updating existing
                tlist = transforms_list
        train_transforms = transforms.Compose(tlist)
        return train_transforms

    @abc.abstractmethod
    def test_transform(self, transforms_list=[], update=True):
        # if no input transforms, use existing
        tlist = self.transforms_list
        if transforms_list:
            if update:
                # append input transforms to existing list
                tlist = self.transforms_list + transforms_list
            else:
                # use input transforms as is without updating existing
                tlist = transforms_list
        test_transforms = transforms.Compose(tlist)
        return test_transforms


def show_images(data_loader, mean, std, classes, number=5, ncols=5, figsize=(10, 6)):
    # torch tensors
    images, labels = next(iter(data_loader))

    # selecting random sample of number
    img_list = random.sample(range(1, images.shape[0]), number)

    set_plt_param()
    rows = (number // ncols) + 1

    # make a grid of subplots for images
    axes = make_grid(rows, ncols, figsize)

    # show images in subplot axis
    for idx, label in enumerate(img_list):
        img = unnormalize_chw_image(images[label], mean, std)
        axes[idx].imshow(img, interpolation='bilinear')
        axes[idx].set_title(classes[labels[label]])

    # Hide empty subplot boundaries
    [ax.set_visible(False) for ax in axes[idx + 1:]]
    plt.show()


def make_grid(nrows, ncols=3, title='', figsize=(6.0, 4.0)):
    """

    Functionality to be added

    :param nrows:
    :param ncols:
    :param figsize:
    :return:
    """
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    # if title:
        # fig.suptitle(title)
    axes = ax.flatten()
    return axes


def set_plt_param():
    rc = {"axes.spines.left": False,
          "axes.spines.right": False,
          "axes.spines.bottom": False,
          "axes.spines.top": False,
          "axes.grid": False,
          "xtick.bottom": False,
          "xtick.labelbottom": False,
          "ytick.labelleft": False,
          "ytick.left": False}
    plt.rcParams.update(rc)

