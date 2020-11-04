import random
import os
from PIL import Image,ImageFilter,ImageDraw
from torchvision import datasets, transforms
import numpy as np
import h5py
import sys
from PIL import ImageStat
import cv2

class listDataset():
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        img,target,mask,mask_den = load_data(img_path,self.train)
        transform_mask = transforms.Compose([
            transforms.ToTensor()])
        mask = transform_mask(mask)
        mask_den = transform_mask(mask_den)
        img = self.transform(img)
        return img,target,mask,mask_den

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth').replace('_frame','_label')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    img = np.asarray(img)
    target = cv2.resize(target,(target.shape[1]//8,target.shape[0]//8),interpolation = cv2.INTER_NEAREST)*64
    mask = Image.open('./masks/test/' + img_path.split('/')[-1])
    mask_den = Image.open('./masks/tests/' + img_path.split('/')[-1])
    return img,target,mask,mask_den
