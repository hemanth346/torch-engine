import abc
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from engine.utils import *

__all__ = ['make_grid', 'set_plt_param', 'show_images']


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

