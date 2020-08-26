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
from model_cpu import LSTMModel
import cv2 


PATH="./save_model/model_best.pth.tar"

device="cpu"
original_model=models.alexnet(pretrained=True)
lstm_layers=2
num_classes=2
hidden_layers=512
fc=4096
lr=1e-4
model = LSTMModel(original_model,"alexnet",num_classes,lstm_layers,hidden_layers,fc)
optimizer = torch.optim.Adam([{'params': model.fc_pre.parameters()},
                                {'params': model.rnn.parameters()},
                                {'params': model.fc.parameters()}],lr)

checkpoint=torch.load(PATH)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch=checkpoint['epoch']
model.eval()
# choose one
# model_type = "3dcnn"
model_type = "rnn"    

timesteps =16
if model_type == "rnn":
    h, w =224, 224
    mean = [0.485, 0.456, 0.406]
    std = [0.339, 0.224, 0.225]
else:
    h, w = 112, 112
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]

parser=argparse.ArgumentParser()
parser.add_argument('--path', default="./test", help='video frames to classify, please give the directory path')
args=parser.parse_args()

test_transformer = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ]) 

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.339, 0.224, 0.225])


def predict_image(image):
    image_tensor = test_transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_var = Variable(image_tensor)
    #input_var = input_var.to(device)
    output = model(input_var)
    index = output.data.cpu().numpy().argmax()
    return index


data_dir=args.path
  

def FrameCapture(path):

    print index 
    global S,N
    S=0
    N=0
    cap = cv2.VideoCapture(os.path.join(data_dir,path)) 
    count = 0
    success = 0
    #video_name=(path.split('/'))[-1]
    while(cap.isOpened()): 
  
        success, image = cap.read() 
        if not success:
            break
        pil_image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        index=predict_image(pil_image)
        if index==1:
        	S+=1
        else :
        	N+=1
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print()
    print(f"Smoking %: {S/count:.3f}.. "  f"Non Smoking %: {N/count:.3f}.. ")
    print("Predicting...\n")
    threshold = 0.8
    if S >= threshold:
        print("Predicted: Smoking")
    elif N >= threshold:
        print("Predicted: Non Smoking")
    else:
        print("Not predicted, clarify")
    

for video_name in os.listdir(data_dir):
    print(video_name)
    FrameCapture(video_name)




'''
SCAM
data_transforms = {
    'predict': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }
dataset = {'predict' : datasets.ImageFolder("./dataset/frames/", data_transforms['predict'])}
    dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = 1)}

    outputs = list()
    since = time.time()
    for inputs, labels in dataloader['predict']:
        inputs = inputs.to(device)
        output = model(inputs)
        output = output.to(device)
        index = output.data.numpy().argmax()
    print(index)
'''