from engine import datasets
import torch
import torchvision
import torchvision.transforms as transforms

# logger = setup_logger(__name__)
"""
### Dataset overview
We have 
 - 2 flat files 
    - one with list of class ids in this dataset and 
    - another with labels of the class ids, but for a superset
   
   
  - 3 folders one for each 
    - train
     - has **200** folders - one for each class_id. 
     - Each class folder has 
       -  `images` folder with **500** JPEG images named in the format `<class_id>_serialno.JPEG`
       - flat file `<class_id>_boxes.txt`with column names as 
         - filename of the image
         - the co-ordinates(x, y, h, w) 
    
    - validation    
      - `images` folder with **10,000** JPEG images named as val_S.no
      - flat file `val_annotations.txt` with columsn as 
        - file name, 
        - corresponding class, 
        - the co-ordinates(x, y, h, w)
   
    - test
      - *images* folder with **10000** JPEG images named as test_S.no
      

### Stats
Dataset has 
```
    - 200 classes
    - 100,000 traning images(200*500)
    - 10,000 validation images
    - 10,000 test images
```
"""

__all__ = ['mean', 'std', 'classes', 'transforms_list', 'dataloader_args', 'data_loaders', 'show_images']

# Constants
# Constants
mean = (0.4802, 0.4481, 0.3975)
std = (0.2302, 0.2265, 0.2262)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transforms_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
]

# default dataloader_args for all datasets, available to modify as attribute
dataloader_args: dict = datasets.dataloader_args



def get_labels(data_dir):
    classes = [line.strip() for line in open(data_dir + 'wnids.txt', 'r')]
    class_ids = {cls: idx for idx, cls in enumerate(classes)}

    all_labels = {}
    with open(data_dir + 'words.txt', 'r') as f:
        for idx, word in enumerate(f):
            wnid, labels = word.split('\t')
            all_labels[wnid] = labels.replace('\n', '')

    class_labels = {key: all_labels[key] for key in class_ids.keys()}
    return class_ids, class_labels


def data_loaders(batch_size=64, augmentations=[]):
    """

    :param batch_size:
    :param augmentations:
    :return: Data loaders
    """
    train_transform_list = transforms_list

    if augmentations:
        train_transform_list = [augmentations] + transforms_list

    train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose(transforms_list)

    # Get the Train and Test Set
    trainset = None
        # torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    testset = None
        # torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

    dataloader_args['batch_size'] = batch_size

    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return train_loader, test_loader


def show_images(data_loader, number=5, ncols=5, figsize=(10, 6)):
    datasets.show_images(data_loader, mean, std, classes, number, ncols, figsize)
