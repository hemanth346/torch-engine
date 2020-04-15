import sys
import logging
import numpy as np

# https://stackoverflow.com/a/64130/7445772
__all__ = ['setup_logger', 'AlbumentationToPytorchTransform', 'unnormalize_chw_image', 'unnormalize_hwc_image', 'UnNormalize']


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
