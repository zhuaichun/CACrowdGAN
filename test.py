import h5py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision import models
from dataload import listDataset
from model import Dense, CSRNet
import PIL
import shutil
import numpy as np
import argparse
import json
import torch.nn.functional as F
import cv2
import time
import warnings
from PIL import Image
import os
import random
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import cm as CM
width = 1024
height = 768

val_data_path = '../part_A/test_data/images/'
val_list = []
for i in os.listdir(val_data_path):
    val_list.append(val_data_path + i)

weight_path = './CSRNet_model_bestB.pth.tar'
density_model = False
if density_model:
    model = Dense()
else:
    model = CSRNet()
att = CSRNet()
model = model.cuda()
att = att.cuda()

checkpoint = torch.load(weight_path)
model.load_state_dict(checkpoint['density_state_dict'])
att.load_state_dict(checkpoint['attention_state_dict'])


test_loader = torch.utils.data.DataLoader(
listDataset(val_list,
                 shuffle=False,
                 transform=transforms.Compose([
                     transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
                 ]),  train=False),
batch_size=1)

att.eval()
model.eval()


for i in os.listdir(val_data_path):
    img = Image.open(val_data_path + i).convert('RGB')
    transform=transforms.Compose([
            transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]),])
    img = transform(img).unsqueeze(0)
    img = img.cuda()
    img = Variable(img)
    output = att(img).squeeze()
    output = torch.gt(output, torch.Tensor([0.0]).cuda()).type(torch.cuda.FloatTensor)
    output = output.cpu().numpy()
    cv2.imwrite('./masks/tests/' + i, output)
    output = cv2.resize(output,(img.shape[3],img.shape[2]),interpolation = cv2.INTER_NEAREST)
    cv2.imwrite('./masks/test/' + i, output)



mae = 0
with torch.no_grad():    
    for i,(img, target, mask, mask_den) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        img = torch.mul(img, (mask > 0.).type(torch.FloatTensor).squeeze().cuda())
        output = model(img)
        output = torch.mul(output, (mask_den > 0.).type(torch.FloatTensor).squeeze().cuda()) 
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
        print(output.data.sum(), target.sum().type(torch.FloatTensor).cuda())
        
mae = mae/len(test_loader) 
print(len(test_loader))   
print(' * MAE {mae:.3f} '
          .format(mae=mae))
