import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
import torch

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    Read SAR and optical pairs from trainA/trainB or testA/testB
    under the configured dataset root.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        if not self.A_paths or len(self.A_paths) != len(self.B_paths):
            raise ValueError(f'Paired dataset must be nonempty with equal counts: A={len(self.A_paths)}, B={len(self.B_paths)}')
        if getattr(opt, 'pair_by_name', False):
            a_names = {os.path.relpath(p, self.dir_A): p for p in self.A_paths}
            b_names = {os.path.relpath(p, self.dir_B): p for p in self.B_paths}
            if a_names.keys() != b_names.keys():
                raise ValueError(f'Unpaired images: A-only={sorted(a_names.keys() - b_names.keys())[:5]}, B-only={sorted(b_names.keys() - a_names.keys())[:5]}')
            keys = sorted(a_names)
            self.A_paths = [a_names[k] for k in keys]
            self.B_paths = [b_names[k] for k in keys]
        limit = min(len(self.A_paths), opt.max_dataset_size)
        self.A_paths = self.A_paths[:limit]
        self.B_paths = self.B_paths[:limit]
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        with Image.open(A_path) as image:
            A = image.convert('RGB')
        with Image.open(B_path) as image:
            B = image.convert('RGB')
        if A.size != B.size:
            raise ValueError(f'Paired image sizes differ: {A_path} {A.size}, {B_path} {B.size}')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""

        return len(self.A_paths)

    def random_crop(self, sar, rgb, crop_size):
        h = np.random.randint(0, sar.shape[1] - crop_size)
        w = np.random.randint(0, sar.shape[2] - crop_size)
        sar = sar[:, h:h + crop_size, w:w + crop_size]
        rgb = rgb[:, h:h + crop_size, w:w + crop_size]
        return sar, rgb

    def center_crop(self, sar, rgb, crop_size):
        sar = sar[:, 194:194 + crop_size, 194:194 + crop_size]
        rgb = rgb[:, 194:194 + crop_size, 194:194 + crop_size]
        return sar, rgb









