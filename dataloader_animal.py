from torch.utils.data import Dataset, DataLoader
from randaugment import RandAugment
import copy
import torchvision.transforms as transforms
import os 
import random
import numpy as np
from PIL import Image
import json
import torch
# https://github.com/kuangliu/pytorch-retinanet/blob/master/transform.py
def resize(img, size, max_size=1000):
    '''Resize the input PIL image to the given size.
    Args:
      img: (PIL.Image) image to be resized.
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w, h)
        sw = sh = float(size) / size_min

        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow, oh), Image.BICUBIC)


# labeled_dataset = animal_dataset(self.root,transform=self.transform_train, mode='labeled',pred=pred, probability=prob,paths=paths)
class TransformTwice:
    def __init__(self, transform1,transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        return out1, out2
class animal_dataset(Dataset):
    def __init__(self, data_path, mode, transform,pred=None, probability=None,paths=[]):

        self.pred=pred
        self.probability=probability
        self.paths=paths
        self.mode=mode
        if mode=="all":
            self.image_dir = os.path.join(data_path, 'training')
        elif mode=='labeled':
            self.image_dir = os.path.join(data_path, 'training')
        elif mode=='test':
            self.image_dir = os.path.join(data_path, 'testing')

        self.image_files = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]

        self.targets = []

        for path in self.image_files:
            label = path.split('_')[0]
            self.targets.append(int(label))

        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])

        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        label = self.targets[index]
        label = np.array(label).astype(np.int64)

        if self.mode=='labeled':
            return image, torch.from_numpy(label),self.probability[index], index
        else:
            return image, torch.from_numpy(label), index

    def __len__(self):
        return len(self.targets)

class animal_dataloader():  
    def __init__(self, root, batch_size, num_workers):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
                   
        self.transform_train = transforms.Compose([
                # transforms.Resize(256),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 
        self.transform_test = transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])        
        self.strong_transform = copy.deepcopy(self.transform_train)
        self.strong_transform.transforms.insert(0, RandAugment(3,5))

    def run(self,mode,pred=[],prob=[],paths=[]):        
        if mode=='warmup':
            warmup_dataset = animal_dataset(self.root,transform=self.transform_train, mode='all')
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader
        elif mode=='train':
            labeled_dataset = animal_dataset(self.root,transform=TransformTwice(self.transform_train,self.strong_transform,), mode='labeled',pred=pred, probability=prob,paths=paths)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)           
            return labeled_loader
        elif mode=='eval_train':
            eval_dataset = animal_dataset(self.root,transform=self.transform_train, mode='all')
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        
        elif mode=='test':
            test_dataset = animal_dataset(self.root,transform=self.transform_test, mode='test',)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=1000,
                shuffle=False,
                num_workers=self.num_workers)             
            return test_loader             


