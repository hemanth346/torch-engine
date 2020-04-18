from torch.nn import functional as F
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from engine.utils import UnNormalize as unnormalize

# https://github.com/kazuto1011/grad-cam-pytorch/blob/fd10ff7fc85ae064938531235a5dd3889ca46fed/grad_cam.py


class GradCAM:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers
    target_layers = list of convolution layer index as shown in summary
    """

    def __init__(self, model, candidate_layers=None):
        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.nll).to(self.device)
        print(one_hot.shape)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]  # HxW
        self.nll = self.model(image)
        # self.probs = F.softmax(self.logits, dim=1)
        return self.nll.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.nll.backward(gradient=one_hot, retain_graph=True)

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        # need to capture image size duign forward pass
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        # scale output between 0,1
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def GRADCAM(images, labels, learner, target_layers):
    learner.model.eval()
    # map input to device
    images = torch.stack(images).to(learner.device)
    # set up grad cam
    gcam = GradCAM(learner.model, target_layers)
    # forward pass
    probs, ids = gcam.forward(images)
    # outputs agaist which to compute gradients
    ids_ = torch.LongTensor(labels).view(len(images), -1).to(learner.device)
    # backward pass
    gcam.backward(ids=ids_)
    layers = []
    for i in range(len(target_layers)):
        target_layer = target_layers[i]
        print("Generating Grad-CAM @{}".format(target_layer))
        # Grad-CAM
        layers.append(gcam.generate(target_layer=target_layer))
    # remove hooks when done
    gcam.remove_hook()
    return layers, probs, ids


# http://jonathansoma.com/lede/data-studio/classes/small-multiples/long-explanation-of-using-plt-subplots-to-create-small-multiples/
# https://napsterinblue.github.io/notes/python/viz/subplots/
# https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html

def PLOT(gcam_layers, images, labels, target_layers, class_names, image_size, predicted, output_size=(128, 128)):
    # https://stackoverflow.com/a/53721862/7445772
    # https://matplotlib.org/tutorials/introductory/customizing.html
    rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "axes.grid" : False,
      "xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}
    plt.rcParams.update(rc)

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
