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
from threading import Thread
from time import sleep

from os import system, name
import sys
import Display_video as Dv

from tkinter import *
parser=argparse.ArgumentParser()
parser.add_argument('--frames_dir', default="./test", help='video frames to classify, please give the directory path')
parser.add_argument('--model',default = "smoking.pth", help = 'please give the path to the trained model')
args=parser.parse_args()

FILEPATH = args.model

#model=alexnet()
model = torch.load(FILEPATH)
model.eval()

test_transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()])
            
data_dir=args.frames_dir

'''
def test(input, model, eval_ratio=0.5):
	eval_last_len = int(len(input) * eval_ratio)

	#model.eval()
	input_var = Variable(input)
	#input_var = input_var.cuda()
	outputs= model(input_var)
	weight = Variable(torch.Tensor(range(outputs.shape[0])) / (outputs.shape[0] - 1) * 2)
	output = torch.mean(outputs * weight.unsqueeze(1), dim=0)
	#output = nn.functional.softmax(output, dim=0)
	#print(output)

	confidence, idx = torch.max(output.data.cpu(), 0)
	#print(confidence,idx)
	return confidence.numpy()[0], idx.numpy()[0]
'''
import cv2 

# Function to extract frames 
def FrameCapture(path): 
    global S,N
    S=0
    N=0
    cap = cv2.VideoCapture(os.path.join(data_dir,path)) 
    count = 0
    success = 0
    while(cap.isOpened()): 
  
        success, image = cap.read() 
        #cv2.imshow(image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #im_pil = Image.fromarray(image)
        if not success:
            break
        pil_image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        index=predict_image(pil_image)
        #print(index)
        if index==1:
        	S+=1
        else :
        	N+=1
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    
    print()
    print("Smoking %: {:.3f}.. Non Smoking %: {:.3f}.. ".format(S/count,N/count))
    print()
    print("Predicting...\n")
    if S>N :
        print("Predicted: Smoking")
    else :
        print("Predicted: Non Smoking")
    cv2.destroyAllWindows()
    
    #sleep(1)
def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_var = Variable(image_tensor)
    output = model(input_var)
    #print(output)
    index = output.data.cpu().numpy().argmax()
    return index




       


    
   

def clea():
    os.system("printf '\033c'")
    

    
for video_name in os.listdir(data_dir):
    
    print(video_name) 
    dirp='/home/hduser/Bai-labs/bai_labs-master2/Smoking_detection_Cmd_Line_v2/test1/'
    vidpth=os.path.join(dirp,video_name)
    obj=Dv.videoGUI(vidpth)
    
    Thread(target =obj ).start()
    #sleep(0.2)
    del obj
    Thread(target =FrameCapture(video_name)).start()
    #t1.join()
    #t2.join()
    #FrameCapture(video_name)
    sleep(0.1)
    print("press any key to continue")
    input()
    clea()

