import random

import numpy as np
from matplotlib import pyplot as plt

from engine.utils import unnormalize_chw_image, get_misclassified, UnNormalize as unnormalize
from engine.utils.plots.helper import show_metrics, show_lr_metrics, make_grid, set_image_plt_param


def show_images(data_loader, mean, std, classes, number=5, ncols=5, figsize=(10, 6)):
    # torch tensors
    images, labels = next(iter(data_loader))

    # selecting random sample of number
    img_list = random.sample(range(1, images.shape[0]), number)

    set_image_plt_param()
    rows = (number // ncols) + 1

    # make a grid of subplots for images
    fig, axes = make_grid(rows, ncols, figsize)

    # show images in subplot axis
    for idx, label in enumerate(img_list):
        img = unnormalize_chw_image(images[label], mean, std)
        axes[idx].imshow(img, interpolation='bilinear')
        axes[idx].set_title(classes[labels[label]])

    # Hide empty subplot boundaries
    [ax.set_visible(False) for ax in axes[idx + 1:]]
    plt.show()


def plot_history(model, metrics=['train_acc', 'train_loss', 'val_acc', 'val_loss'], show_lr=False, nplots=2):
    data = []
    for metric in metrics:
        data.append(model.metrics[metric])
    if show_lr:
        show_lr_metrics(data, metrics, lr_history=model.metrics['lr'], nplots=nplots)
    else:
        show_metrics(data, metrics, nplots)


def show_misclassified_images(model, data_loader, classes, mean, std, number=20, device='cpu', ncols=5, figsize=(10, 6)):
    set_image_plt_param()
    images, ground_truth, predicted = get_misclassified(model=model, data_loader=data_loader,
                                                        number=number, device=device)
    rows = (number // ncols) + 1
    # make a grid of subplots for images
    fig, axes = make_grid(rows, ncols, figsize=figsize)

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



def show_gradcam(gcams, images, ground_truth, predicted, class_names, predicted, output_size=(128, 128)):
    set_image_plt_param()
    # axes = make_grid()

    # images, ground_truth, predicted = get_misclassified(model=model, data_loader=data_loader,
    #                                                     number=number, device=device)

    nrows = len(images)
    ncols = len(gcams.keys()+1)
    # make a grid of subplots for images
    fig, axes = make_grid(rows=nrows, ncols=ncols, figsize=(nrows*2, ncols*2))
    fig.suptitle('Gradient Class Activation Mappings(GradCAM)')

    # show images in subplot axis
    for idx in range(ncols):
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

    for idx in range()
    pass


# http://jonathansoma.com/lede/data-studio/classes/small-multiples/long-explanation-of-using-plt-subplots-to-create-small-multiples/
# https://napsterinblue.github.io/notes/python/viz/subplots/
# https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html
def PLOT(gcam_layers, images, labels, target_layers, class_names, image_size, predicted, output_size=(128, 128)):

    rows = len(images)
    cols = len(target_layers) + 2 # label and input + layers names

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5*rows, 4*cols))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    ax = axes.flatten()

    for image_no in range(rows):
        col1 = image_no*cols

        img = np.uint8(255 * unnormalize(images[image_no].view(image_size)))
        #label
        ax[col1].text(0, 0.2, f"pred={class_names[predicted[image_no][0]]}\n[actual={class_names[labels[image_no]]}]", fontsize=27)
        # 'input_image'

        for layer_no in range(len(target_layers)):
            heatmap = 1 - gcam_layers[layer_no][image_no].cpu().numpy()[0]  # reverse the color map
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.resize(cv2.addWeighted(img, 0.5, heatmap, 0.5, 0), output_size)
            ax[col1 + 2 +layer_no].imshow(superimposed_img, interpolation='bilinear')
        # display after resizing
        img = cv2.resize(img, output_size)
        ax[col1+1].imshow(img, interpolation='bilinear')

    plt.show()




def plot_gradcam(gcam_layers, images, target_labels, predicted_labels, class_labels, denormalize, paper_cmap=False):

    # convert BCHW to BHWC for plotting stufffff

    fig, axs = plt.subplots(nrows=len(images), ncols=len(
        gcam_layers.keys())+2, figsize=((len(gcam_layers.keys()) + 2)*3, len(images)*3))
    fig.suptitle("Grad-CAM", fontsize=16)

    for image_idx, image in enumerate(images):

        # denormalize the imaeg
        denorm_img = denormalize(image.permute(2, 0, 1)).permute(1, 2, 0)

        axs[image_idx, 0].text(
            0.5, 0.5, f'predicted: {class_labels[predicted_labels[image_idx][0] ]}\nactual: {class_labels[target_labels[image_idx]] }', horizontalalignment='center', verticalalignment='center', fontsize=14, )
        axs[image_idx, 0].axis('off')

        axs[image_idx, 1].imshow(
            (denorm_img.numpy() * 255).astype(np.uint8),  interpolation='bilinear')
        axs[image_idx, 1].axis('off')

        for layer_idx, layer_name in enumerate(gcam_layers.keys()):
            # gets H X W of the cam layer
            _layer = gcam_layers[layer_name][image_idx].cpu().numpy()[0]
            heatmap = 1 - _layer
            heatmap = np.uint8(255 * heatmap)
            heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superimposed_img = cv2.addWeighted(
                (denorm_img.numpy() * 255).astype(np.uint8), 0.6, heatmap_img, 0.4, 0)

            axs[image_idx, layer_idx +
                2].imshow(superimposed_img, interpolation='bilinear')
            axs[image_idx, layer_idx+2].set_title(f'layer: {layer_name}')
            axs[image_idx, layer_idx+2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, wspace=0.2, hspace=0.2)
    plt.show()
