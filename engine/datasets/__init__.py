import torch

attributes = ['cuda', 'seed', 'shuffle', 'num_workers', 'pin_memory', 'dataloader_args']
__all__ = ['show_images', 'make_grid', 'set_plt_param'] + attributes

# CUDA?
cuda = torch.cuda.is_available()

# Other User inputs
seed = None
if seed:
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


# cuda attributes
num_workers = 0
pin_memory = True

# dataloader cuda/cpu args
shuffle = True
if cuda:
    dataloader_args = {
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
else:
    dataloader_args = {
        'shuffle': shuffle
    }



