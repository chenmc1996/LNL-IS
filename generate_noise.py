import os
import random
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def waterbirds(r=0.2):
    dir='../data/waterbird_complete95_forest2water2/'
    metadata_df = pd.read_csv( os.path.join(dir, 'metadata.csv'))

    split_dict = {
        'train': 0,
        'val': 1,
        'test': 2
    }
    mask=metadata_df['split']==split_dict['train']
    y_array = metadata_df['y'][mask].values

    L=y_array.shape[0]
    noise_label = []
    idx = list(range(L))

    random.shuffle(idx)
    num_noise = int(r*L)            
    noise_idx = idx[:num_noise]
    print(noise_idx)
    for i in range(L):
        if i in noise_idx:
            noiselabel = 1-y_array[i]
            noise_label.append(noiselabel)
        else:    
            noise_label.append(y_array[i])   
    metadata_df['y'][mask]=noise_label

    metadata_df.to_csv(os.path.join(dir, f'metadata_{r}r.csv') ,index=False)

def celebA(r=0.2):
    dir='../data/celebA/'
    metadata_df  = pd.read_csv( os.path.join(dir, 'list_attr_celeba.csv'))

    split_dict = {
        'train': 0,
        'val': 1,
        'test': 2
    }

    split_df = pd.read_csv(os.path.join(dir, 'list_eval_partition.csv'))
    mask = split_df['partition']==split_dict['train']

    y_array = metadata_df['Blond_Hair'][mask].values

    L=y_array.shape[0]
    noise_label = []
    idx = list(range(L))

    random.shuffle(idx)
    num_noise = int(r*L)            
    noise_idx = idx[:num_noise]
    print(noise_idx)
    for i in range(L):
        if i in noise_idx:
            noiselabel = -y_array[i]
            noise_label.append(noiselabel)
        else:    
            noise_label.append(y_array[i])   
    metadata_df['Blond_Hair'][mask]=noise_label

    metadata_df.to_csv(os.path.join(dir,f'list_attr_celeba_{r}r.csv') ,index=False)

waterbirds(0.25)
waterbirds(0.3)
waterbirds(0.35)
waterbirds(0.4)