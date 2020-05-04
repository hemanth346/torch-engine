import torch
from torch.nn import functional as F


# inspired from
# https://github.com/kazuto1011/grad-cam-pytorch/blob/fd10ff7fc85ae064938531235a5dd3889ca46fed/grad_cam.py


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()

        # assuming all parameters are on the same device
        self.device = next(model.parameters()).device

        # save the model
        self.model = model

        # a set of hook function handlers
        self.handlers = []

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        # get H X W
        self.image_shape = image.shape[2:]

        # apply the model
        self.logits = self.model(image)

        # get the loss by converging along all the channels, dim = CHANNEL
        # sum along CHANNEL is going to be 1, softmax does that
        self.probs = F.softmax(self.logits, dim=1)

        # ordered results
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        '''Class-specific backpropagation'''

        # convert the class id to one hot vector
        one_hot = self._encode_one_hot(ids)

        # zero out the gradients
        self.model.zero_grad()

        # calculate the gradient wrt to the class activations
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        '''Remove all the forward/backward hook functions'''
        for handle in self.handlers:
            handle.remove()


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(
                    module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(
                    module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError(f'Invalid layer name: {target_layer}')

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        # rescale features between 0 and 1
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def get_gradcam(images, labels, model, device, target_layers):
    # move the model to device
    model.to(device)

    # set the model in evaluation mode
    model.eval()

    # get the grad cam
    gcam = GradCAM(model=model, candidate_layers=target_layers)

    # images = torch.stack(images).to(device)

    # predicted probabilities and class ids
    pred_probs, pred_ids = gcam.forward(images)

    # actual class ids
    # target_ids = torch.LongTensor(labels).view(len(images), -1).to(device)
    target_ids = labels.view(len(images), -1).to(device)

    # backward pass wrt to the actual ids
    gcam.backward(ids=target_ids)

    # we will store the layers and correspondings images activations here
    layer_activations = {}

    # fetch the grad cam layers of all the images
    for target_layer in target_layers:
        # logger.info(f'generating Grad-CAM for {target_layer}')

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        layer_activations[target_layer] = regions

    # we are done here, remove the hooks
    gcam.remove_hook()

    # gcam_layers, predicted probability, predicted class_ids
    return layer_activations, pred_probs, pred_ids





# # get the generated grad cam
# gcam_layers, predicted_probs, predicted_classes = get_gradcam(
#     data, target, self.trainer.model, self.trainer.device, target_layers)
#
# # get the denomarlization function
# unorm = module_aug.UnNormalize(
#     mean=self.transforms.mean, std=self.transforms.std)
#
# plot_gradcam(gcam_layers, data, target, predicted_classes,
#              self.data_loader.class_names, unorm)

