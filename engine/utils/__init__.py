import sys
import logging
import numpy as np
import torch
# https://stackoverflow.com/a/64130/7445772
# from torch import __init__

__all__ = ['classwise_accuracy', 'get_misclassified', 'setup_logger', 'AlbumentationToPytorchTransform', 'unnormalize_chw_image', 'unnormalize_hwc_image', 'UnNormalize']


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


def classwise_accuracy(model, test_loader, classes, device='cpu'):
    '''
        Class wise total accuracy
    :param classes:
    :return:
    '''
    class_total = list(0. for i in range(10))
    class_correct = list(0. for i in range(10))

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]:<10} : {(100 * class_correct[i] / class_total[i]):.2f}%')


def get_misclassified(model, data_loader, number=20, device='cpu'):
    # predict_generator
    '''
        Generates output predictions for the input samples.
        predict(x, batch_size=None, verbose=0, steps=None, callbacks=None,
        max_queue_size=10, workers=1, use_multiprocessing=False)
    '''

    misclassified_data = []
    misclassified_ground_truth = []
    misclassified_predicted = []
    model.eval()

    count = 0
    with torch.no_grad():
        for data, target in data_loader:
            # move to respective device
            data, target = data.to(device), target.to(device)
            # inference
            output = model(data)

            # get predicted output and the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            # get misclassified list for this batch
            misclassified_list = (target.eq(pred.view_as(target)) == False)
            misclassified = data[misclassified_list]
            predicted = pred[misclassified_list]
            ground_truth = target[misclassified_list]
            count += misclassified.shape[0]
            # stitching together
            misclassified_data.append(misclassified)
            misclassified_ground_truth.append(ground_truth)
            misclassified_predicted.append(predicted)
            # stop after enough false positives
            if count >= number:
                break

    # converting to torch
    # clipping till given number if more count from batch
    misclassified_data = torch.cat(misclassified_data)[:number]
    misclassified_ground_truth = torch.cat(misclassified_ground_truth)[:number]
    misclassified_predicted = torch.cat(misclassified_predicted)[:number]

    # can't convert CUDA tensor to numpy during image unnorm
    return misclassified_data.cpu(), misclassified_ground_truth.cpu(), misclassified_predicted.cpu()