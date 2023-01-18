
from PIL import Image
from pathlib import Path

from typing import List, Tuple, Sequence, Optional

import torch
from pdb import set_trace
from torchvision import transforms as t
from torchvision.transforms.transforms import ToPILImage

from torch import Tensor
from torchvision.transforms.functional import crop


class TensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):            
        return self.tensors[0].size(0)


###############################
# Follows adapted from avalanche: https://avalanche.continualai.org/


def default_image_loader(path):
    """
    Sets the default image loader for the Pytorch Dataset.

    :param path: relative or absolute path of the file to load.

    :returns: Returns the image as a RGB PIL image.
    """
    return Image.open(path).convert("RGB")


class PathsDataset(torch.utils.data.Dataset):
    """
    This class extends the basic Pytorch Dataset class to handle list of paths
    as the main data source.
    """

    def __init__(
        self,
        root,
        files,
        transform=None,
        target_transform=None,
        loader=default_image_loader,
    ):
        """
        Creates a File Dataset from a list of files and labels.

        :param root: root path where the data to load are stored. May be None.
        :param files: list of tuples. Each tuple must contain two elements: the
            full path to the pattern and its class label. Optionally, the tuple
            may contain a third element describing the bounding box to use for
            cropping (top, left, height, width).
        :param transform: eventual transformation to add to the input data (x)
        :param target_transform: eventual transformation to add to the targets
            (y)
        :param loader: loader function to use (for the real data) given path.
        """

        if root is not None:
            root = Path(root)

        self.root: Optional[Path] = root
        self.imgs = files
        self.targets = [img_data[1] for img_data in self.imgs]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Returns next element in the dataset given the current index.

        :param index: index of the data to get.
        :return: loaded item.
        """

        img_description = self.imgs[index]
        impath = img_description[0]
        target = img_description[1]
        bbox = None
        if len(img_description) > 2:
            bbox = img_description[2]

        if self.root is not None:
            impath = self.root / impath
        img = self.loader(impath)

        # If a bounding box is provided, crop the image before passing it to
        # any user-defined transformation.
        if bbox is not None:
            if isinstance(bbox, Tensor):
                bbox = bbox.tolist()
            img = crop(img, *bbox)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Returns the total number of elements in the dataset.

        :return: Total number of dataset items.
        """

        return len(self.imgs)


class Subset(torch.utils.data.Dataset):
    """
    subset with class mapping
    """
    def __init__(self, dataset, indices, class_mapping, task_labels, transform=None):
        self._dataset = dataset
        self._indices = indices
        self._subset = torch.utils.data.Subset(dataset, indices)
        self._class_mapping = class_mapping
        self._task_labels = task_labels
        self._transform = transform

    def __getitem__(self, index):
        x, y = self._subset[index]
        if self._transform is not None:
            x = self._transform(x)
        mapped_y = self._class_mapping[y]
        return x, mapped_y

    def __len__(self):
        return len(self._indices)

    def get_task_label(self, index):
        if type(self._task_labels) is int:
            return self._task_labels
        return self._task_labels[index]


class Benchmark:
    """
    Benchmark of all experiments
    """
    def __init__(self, train_datasets, test_datasets, val_datasets):
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.val_datasts = val_datasets

