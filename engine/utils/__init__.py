import sys
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
# https://stackoverflow.com/a/64130/7445772
__all__ = ['show_images', 'setup_logger', 'AlbumentationToPytorchTransform', 'unnormalize_chw_image', 'unnormalize_hwc_image', 'UnNormalize', 'make_grid', 'set_plt_param']


class AlbumentationToPytorchTransform():
    """
    Helper class to convert Albumentations compose
    into compatible for pytorch transform compose
    """

    def __init__(self, compose=None):
        #albumentation's compose
        self.transform = compose

    def __call__(self, img):
        img = np.array(img)
        return self.transform(image=img)['image']


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def unnormalize_hwc_image(image, mean, std):
    '''
    In torch Images are stored as shape [BATCH B, CHANNEL C, HEIGHT H, WIDTH W]
    As per discussions in pytorch forums, torch only supports NCHW format, can be changed using permute

    - Using permute method to swap the axis from HWC to CHW
    - Un-normalizes the image
    - Transpose/permute back CHW to HWC for imshow
    - Convert to np int array and multiple by 255

    :param image:
    :param mean:
    :param std:

    :return: unnormalized image as ndarray
    '''
    unorm = UnNormalize(mean=mean, std=std)
    # HWC to CHW and un-norm
    image = unorm(image.permute(2, 0, 1))
    # CHW to HWC for plots
    image = image.permute(1, 2, 0)
    return (image.numpy() * 255).astype(np.uint8)


def unnormalize_chw_image(image, mean, std):
    '''
    In torch Images are stored as shape [BATCH B, CHANNEL C, HEIGHT H, WIDTH W]

    - Un-normalizes the image
    - Transpose/permute back CHW to HWC for imshow
    - Convert to np int array and multiple by 255

    :param image:
    :param mean:
    :param std:

    :return: unnormalized image as ndarray
    '''
    unorm = UnNormalize(mean=mean, std=std)
    # CHW and un-norm
    image = unorm(image)
    # CHW to HWC for plots
    image = image.permute(1, 2, 0)
    return (image.numpy() * 255).astype(np.uint8)


def setup_logger(name, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)  # set the logging level

    # logging format
    logger_format = logging.Formatter(
        '[ %(asctime)s - %(name)s ] %(levelname)s: %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logger_format)

    # file_handler = logging.FileHandler(sys.stdout)
    # file_handler.setFormatter(logger_format)

    logger.addHandler(stream_handler)
    # logger.addHandler(file_handler)
    logger.propagate = False
    return logger  # return the logger


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


def show_misclassified_images(images, ground_truth, predicted, classes, mean, std, ncols=5, figsize=(10, 6)):
    set_plt_param()
    number = images.shape[0]
    rows = (number // ncols) + 1

    # make a grid of subplots for images
    axes = make_grid(rows, ncols, figsize)

    # show images in subplot axis
    for idx in range(number):
        # print('Image number  : ', idx, images[idx].shape)
        img = unnormalize_chw_image(images[idx], mean, std)
        axes[idx].imshow(img, interpolation='bilinear')
        title = f'Ground truth : {classes[ground_truth[idx]]} \nPredicted : {classes[predicted[idx]]}'
        axes[idx].set_title(title)

    # Hide empty subplot boundaries
    [ax.set_visible(False) for ax in axes[idx + 1:]]
    plt.show()

