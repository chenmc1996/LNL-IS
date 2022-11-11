from logging import root
import os
import pickle
from re import sub
from unicodedata import name
from randaugment import RandAugment
import copy
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


model_attributes = {
    'bert': {
        'feature_type': 'text'
    },
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet34': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': False
    },
    'raw_logistic_regression': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': True,
    }
}


def get_transform_celebA(model_type, train):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    if model_attributes[model_type]['target_resolution'] is not None:
        target_resolution = model_attributes[model_type]['target_resolution']
    else:
        target_resolution = (orig_w, orig_h)

    if not train:
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:

        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


class CUBDataset(Dataset):
    def __init__(self, root_dir,  r):

        self.root_dir = root_dir
        self.r = r
        group_array = []
        y_array = []
        self.conf = None
        self.set_meta_data()
        self.set_transform()

    def set_meta_data(self):
        self.model_type = 'resnet50'
        self.target_name = 'Blond_Hair'
        self.confounder_names = ['Male']

        if self.r is None:
            self.attrs_df = pd.read_csv(os.path.join(
                self.root_dir, f'list_attr_celeba.csv'))
        else:
            self.attrs_df = pd.read_csv(os.path.join(
                self.root_dir, f'list_attr_celeba_{self.r}r.csv'))

        self.data_dir = os.path.join(
            self.root_dir, 'img_align_celeba', 'img_align_celeba')
        self.filename_array = self.attrs_df['image_id'].values
        self.attrs_df = self.attrs_df.drop(labels='image_id', axis='columns')
        self.attr_names = self.attrs_df.columns.copy()

        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0

        target_idx = self.attr_idx(self.target_name)
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        confounder_id = confounders @ np.power(2,
                                               np.arange(len(self.confounder_idx)))
        self.confounder_array = confounder_id

        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array = (self.y_array*(self.n_groups/2) +
                            self.confounder_array).astype('int')

        self.split_df = pd.read_csv(os.path.join(
            self.root_dir, 'list_eval_partition.csv'))
        self.split_array = self.split_df['partition'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

    def preload(self):

        pass

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def set_transform(self):
        self.train_transform = get_transform_celebA(
            self.model_type, train=True,)
        self.eval_transform = get_transform_celebA(
            self.model_type, train=False,)

        self.strong_transform = copy.deepcopy(self.train_transform)
        self.strong_transform.transforms.insert(0, RandAugment(3, 5))

    def __getitem__(self, idx,):
        y = self.y_array[idx]
        g = self.group_array[idx]

        img_filename = os.path.join(self.data_dir, self.filename_array[idx])
        o_img = Image.open(img_filename).convert('RGB')

        if self.split_array[idx] == self.split_dict['train']:
            img = self.train_transform(o_img)
        elif self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']]:
            img = self.eval_transform(o_img)

        if self.conf is None:
            return img, y, g
        else:
            return [img, self.strong_transform(o_img)], y, g, self.conf[idx]

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x, y, g in self:
            return x.size()

    def set_conf(self, estimated_conf):
        if estimated_conf is None:
            self.conf = None
        else:
            self.conf = np.zeros(
                (self.split_array.shape[0],), dtype=np.float32)

            mask = (self.split_array == self.split_dict['train'])
            self.conf[mask] = estimated_conf

    def get_splits(self, splits):
        subsets = {}
        for split in splits:
            assert split in ('train', 'val', 'test'), split + \
                ' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            subsets[split] = Subset(self, indices)
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name


def prepare_data(args, train, r):
    all_data = CUBDataset(root_dir=args.root_dir, r=r)

    if train:
        splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    all_data_split = all_data.get_splits(splits)
    return all_data, all_data_split


if __name__ == "__main__":
    pass
