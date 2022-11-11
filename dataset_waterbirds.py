import os
from cutout import Cutout
from re import sub
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


def get_transform_cub(train, augment_data=False):
    scale = 256.0/224.0
    target_resolution = model_attributes['resnet50']['target_resolution']
    assert target_resolution is not None

    if (not train) or (not augment_data):

        transform = transforms.Compose([
            transforms.Resize(
                (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


class CUBDataset(Dataset):
    def __init__(self, root_dir,  r, subpopulation):

        self.root_dir = root_dir
        self.r = r
        self.subpopulation = subpopulation
        group_array = []
        y_array = []
        self.conf = None
        self.set_meta_data()
        self.set_transform()

    def set_meta_data(self):
        self.data_dir = os.path.join(self.root_dir, 'data', '_'.join(
            [f'waterbird_complete{self.subpopulation}', 'forest2water2']))
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        if self.r is None:

            self.metadata_df = pd.read_csv(
                os.path.join(self.data_dir, 'metadata.csv'))
        else:
            self.metadata_df = pd.read_csv(
                os.path.join(self.data_dir, f'metadata_{self.r}r.csv'))

        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1

        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) +
                            self.confounder_array).astype('int')

        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

    def set_transform(self):
        scale = 256.0/224.0
        target_resolution = model_attributes['resnet50']['target_resolution']
        self.train_transform = get_transform_cub(
            train=True,
        )
        self.eval_transform = get_transform_cub(
            train=False,
        )

        self.strong_transform = copy.deepcopy(self.train_transform)
        self.strong_transform.transforms.insert(0, RandAugment(3, 5))
        self.cutout_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            Cutout(n_holes=1, length=64)
        ])

    def preload(self):
        self.o_img = []
        for i, filename in enumerate(self.filename_array):
            if i % 1000 == 0:
                print(i)
            img_filename = os.path.join(self.data_dir, filename)
            self.o_img.append(Image.open(img_filename).convert('RGB'))

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
                (self.split_array.shape[0], 2), dtype=np.float32)

            mask = (self.split_array == self.split_dict['train'])
            self.conf[mask] = estimated_conf

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ('train', 'val', 'test'), split + \
                ' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac < 1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(
                    indices)[:num_to_retain])
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

    def get_loader(self, train, reweight_groups, **kwargs):
        if not train:
            assert reweight_groups is None
            shuffle = False
            sampler = None
        elif not reweight_groups:
            shuffle = True
            sampler = None
        else:

            group_weights = len(self)/self._group_counts
            weights = group_weights[self._group_array]

            sampler = WeightedRandomSampler(
                weights, len(self), replacement=True)
            shuffle = False

        loader = DataLoader(
            self,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs)
        return loader


def prepare_data(args, train, r, subpopulation):
    all_data = CUBDataset(root_dir=args.root_dir, r=r,
                          subpopulation=subpopulation)

    if train:
        splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    all_data_split = all_data.get_splits(splits, train_frac=args.fraction)
    return all_data, all_data_split
