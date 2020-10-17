import torch
import argparse
import numpy as np
import os,glob,cv2
from torchvision import datasets, transforms, utils
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler


#CONVERT VIDEOS TO FRAMES

def read_frame(classname,label,path,data_path):
  vid = 0
  for filename in os.listdir(os.path.join(path,classname)):
    vid +=1
    print("%d "%vid,end = " ")
    #count = 0
    cap = cv2.VideoCapture(os.path.join(path,classname,filename))   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    #x=1
    while(cap.isOpened()):
      frameId = cap.get(1) #current frame number
      ret, frame = cap.read()
      if(ret != True):
        break
      if(frameId%15==0):
        imgname ="%d%dframe%d.jpg" %(label,vid,frameId);
        cv2.imwrite(os.path.join(data_path,imgname), frame)
    cap.release()
    

def DataTransform():
    #data transform is correct for 224x224
    data_transform = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }
    return data_transform

#accuracy changed 
def calc_accuracy(model, data,dataloaders,cuda=False):
    total = []
   # model.to(device='cuda')    
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[data]):
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # obtain the outputs from the model
            outputs = model.forward(inputs)
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1)

            equals = predicted == labels.data
            
            batch_acc = equals.float().mean()
            print(batch_acc)
            total.append(batch_acc)
    print("Final Test Accuracy : "+str(np.mean(total)))