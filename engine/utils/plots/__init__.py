import random
from matplotlib import pyplot as plt

from engine.utils import unnormalize_chw_image, get_misclassified
from engine.utils.plots.helper import show_metrics, show_lr_metrics, make_grid, set_image_plt_param


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


def show_misclassified_images(model, data_loader, classes, mean, std, number=20, device='cpu', ncols=5, figsize=(10, 6)):
    set_image_plt_param()
    images, ground_truth, predicted = get_misclassified(model=model, data_loader=data_loader,
                                                        number=number, device=device)
    rows = (number // ncols) + 1
    # make a grid of subplots for images
    axes = make_grid(rows, ncols, figsize=figsize)

    # show images in subplot axis
    for idx in range(number):
        # print('Image number  : ', idx, images[idx].shape)
        img = unnormalize_chw_image(images[idx], mean, std)
        axes[idx].imshow(img, interpolation='bilinear')
        title = f'Ground truth : {classes[ground_truth[idx]]} \nPredicted : {classes[predicted[idx]]}'
        axes[idx].set_title(title, fontsize=11)

    # Hide empty subplot boundaries
    [ax.set_visible(False) for ax in axes[idx + 1:]]
    # to reve unnecesary
    # [ax.remove() for ax in axes[idx + 1:]]
    plt.show()


