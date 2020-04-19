import random
import numpy as np
from matplotlib import pyplot as plt

from engine.utils import unnormalize_chw_image


def show_metrics(data=[[]],labels=[], nplots=1):
    # assert len(data) > nplots
    # assert len(data) == len(labels)
    set_plt_param()
    if nplots == 1:
        # plot all train and test in single plot
        for idx in range(len(data)):
            plt.plot(data[idx], label=labels[idx])
    elif nplots == 2:
        # plot accuracy and loss metrics separately
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Model history')
        axes = axs.flatten()
        for idx in range(len(data)):
            ax_idx = idx % 2
            axes[ax_idx].plot(data[idx], label=labels[idx])
            axes[ax_idx].legend()
    elif nplots == 4:
        # plot all metrics separately
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Model history')
        axes = axs.flatten()
        for idx in range(len(data)):
            axes[idx].plot(data[idx], label=labels[idx])
            axes[idx].legend()
    plt.legend()
    plt.show()


def show_lr_metrics(data=[[]],labels=[], lr_history=None, nplots=1):

    set_plt_param()
    if lr_history:
        epochs = np.arange(1, len(lr_history) + 1)
    # else:
    #     # batches
    #     epochs = np.arange(1, len(data[0]) + 1)

    if nplots == 1:
        # plot all train and test in single plot
        for idx in range(len(data)):
            # failing since the dimensions are not equal
            # plt.plot(epochs, data[idx], label=labels[idx])
            plt.plot(data[idx], label=labels[idx])
            if lr_history:
                plt.plot(epochs, lr_history, label='Learing rate')

    elif nplots == 2:
        # plot accuracy and loss metrics separately
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Model history')

        axes = axs.flatten()
        for idx in range(len(data)):
            ax_idx = idx % 2
            axes[ax_idx].plot(data[idx], label=labels[idx])
            if lr_history:
                # instantiate a second axes that shares the same x-axis
                ax = axes[ax_idx].twinx()
                ax.plot(epochs, lr_history, label='LR')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Learning rate')
                ax.legend()
            axes[ax_idx].legend()

    elif nplots == 4:
        # plot all metrics separately
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Model history')
        axes = axs.flatten()
        for idx in range(len(data)):
            axes[idx].plot(data[idx], label=labels[idx])
            axes[idx].legend()
            if lr_history:
                # instantiate a second axes that shares the same x-axis
                ax = axes[idx].twinx()
                ax.plot(epochs, lr_history, label='LR')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Learning rate')
                ax.legend()
    plt.legend()
    plt.show()





    # plt.rcParams.update({'figure.figsize': (9, 6), 'figure.dpi': 120})
    # fig, ax1 = plt.subplots()
    # t = np.arange(1, 25)
    # color = 'tab:red'
    # ax1.set_xlabel('epoch (s)')
    # ax1.set_ylabel('accuracy', color=color)
    # testline, = ax1.plot(t, model.stats().test_acc, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    #
    # color = 'tab:green'
    # trainline, = ax1.plot(t, model.stats().train_acc, color=color)
    # ax1.legend((trainline, testline), ('Train', 'Test'), loc=7)
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #
    # color = 'tab:blue'
    # ax2.set_ylabel('learning rate', color=color)  # we already handled the x-label with ax1
    # lrline, = ax2.plot(t, model.stats().lr, color=color)
    # ax2.legend((lrline,), ('LR',), loc=8)
    # ax2.tick_params(axis='y', labelcolor=color)
    #
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title("Learning Rate and Train/test Accuracy Comparison")
    # plt.show()



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
    rc = {"axes.spines.left": True,
          "axes.spines.right": True,
          "axes.spines.bottom": True,
          "axes.spines.top": True,
          "axes.grid": True,
          "xtick.bottom": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          "ytick.left": True}
    plt.rcParams.update(rc)

def set_image_plt_param():
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

    set_image_plt_param()
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
    set_image_plt_param()
    number = images.shape[0]
    rows = (number // ncols) + 1
    # can't convert CUDA tensor to numpy during image unnorm
    images, ground_truth, predicted = images.cpu(), ground_truth.cpu(), predicted.cpu()

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