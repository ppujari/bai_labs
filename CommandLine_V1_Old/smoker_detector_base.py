#!/usr/bin/env python
# coding: utf-8
# Imports here
import numpy as np
import argparse
from PIL import Image
import json, time, copy
import cv2,os, glob
import pandas as pd
# import seaborn as sns
from collections import OrderedDict

import torch
from torchvision import datasets, transforms, utils
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

#call functions from files
from utils import *
from models import *
from train import *
'''
print("Imports complete")
#read_frame() 
path = '/home/avi/Desktop/SMokedetector_base/Videos/'
data_path = '/home/avi/Desktop/SMokedetector_base/Videos/Sframes'
read_frame('Smoking_dataset',1,path,data_path)

path = '/home/avi/Desktop/SMokedetector_base/Videos/'
data_path = '/home/avi/Desktop/SMokedetector_base/Videos/Nsframes'
read_frame('Non_smoking_dataset',0,path,data_path)

'''
models = {'resnet': 'resnet18', 'alexnet': 'alexnet', 'vgg': 'vgg19'}
'''
resnet50 = models.resnet50(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg19 = models.vgg19(pretrained=True)
'''
#Arg parse arguments 
#type python smoke_detector_base.py --m alexnet/resnet --eps 15 --lr 0.001
parser=argparse.ArgumentParser()
parser.add_argument('--m',type=str , default='alexnet',help='Models are : alexnet,resnet,vgg')
parser.add_argument('--eps',type=int,default=1,help='# of Epochs')
parser.add_argument('--lr',type=float,default=0.001,help='Learning rate')
in_args = parser.parse_args()


data_transform=DataTransform()
print("Data transforms done")

data_dir = '/home/avi/Desktop/SMokedetector_base/data_new'

dirs = {'train': data_dir + '/Train', 
        'valid': data_dir + '/Valid', 
        'test' : data_dir + '/Test'}

image_datasets = {x: datasets.ImageFolder(dirs[x],transform=data_transform[x]) for x in ['train', 'valid', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes


#model name passed as arguments
model_name = models[in_args.m]
model=getModel(class_names,model_name)

#learning rate
lr=in_args.lr

# Criteria NLLLoss which is recommended with Softmax final layer
#(negative log loss)
criteria = nn.NLLLoss() 
# Observe that all parameters are being optimized

#call model.fc.parameters
#model.classifier is not an attribute for resnet
#do not change
if in_args.m=="resnet":
	optimizer = optim.Adam(model.fc.parameters(), lr)
else:
	optimizer = optim.Adam(model.classifier.parameters(), lr)
# Decay LR by a factor of 0.1 every 4 epochs
sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

# Number of epochs
eps=in_args.eps

model_ft = train_model(model,dataloaders,criteria, optimizer, sched,dataset_sizes,eps)
print("Model trained")

calc_accuracy(model_ft,'test',dataloaders)
print("Accuracy")