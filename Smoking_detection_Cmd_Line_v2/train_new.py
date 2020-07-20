import numpy as np
import argparse
import glob
from PIL import Image
import json, time, copy
import cv2,os, glob
import pandas as pd
from collections import OrderedDict
from torch.autograd import Variable 
import torch
from torchvision import datasets, transforms, utils
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

#from test import *

parser=argparse.ArgumentParser()
parser.add_argument('--frames_dir', default=None, help='video frames to classify, please give the directory path')
parser.add_argument('--eps',type=int,default=1,help='No of Epochs')
parser.add_argument('--lr',type=float,default=0.003,help='Learning rate')
args=parser.parse_args()

#frames=args.frames_dir


#model = models.resnet50(pretrained=True)
#print(model)
data_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()])


model = models.alexnet(pretrained = True)
print(model)


for param in model.parameters():
    param.requires_grad = False


classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(9216, 4096)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 2)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
model.classifier=classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), args.lr)
#model.to(device)


data_dir = '/home/avi/Desktop/SMokedetector_base/data_new'

dirs = {'train': data_dir + '/Train', 
        'valid': data_dir + '/Valid', 
        'test' : data_dir + '/Test'}


image_datasets = {x: datasets.ImageFolder(dirs[x],transform=data_transform) for x in ['train', 'valid', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes


trainloader=dataloaders['train']
testloader=dataloaders['valid']

epochs = args.eps
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], [] 

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        #inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    #inputs, labels = inputs.to(device),
                    #                  labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()


FILEPATH="smoking.pth"
torch.save(model,FILEPATH)

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()
