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
    cv2.destroyAllWindows()
    print()
    print(f"Smoking %: {S/count:.3f}.. "
    	 f"Non Smoking %: {N/count:.3f}.. ")
    print()
    print("Predicting...\n")
    if S>N :
        print("Predicted: Smoking")
    else :
        print("Predicted: Non Smoking")

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_var = Variable(image_tensor)
    output = model(input_var)
    #print(output)
    index = output.data.cpu().numpy().argmax()
    return index



for video_name in os.listdir(data_dir):
    print(video_name)
    FrameCapture(video_name)


'''
def main():
	#global args
	#args = parser.parse_args()
	if args.frames_dir is not None:
			# classify a video 
			frames_list = sorted(os.listdir(args.frames_dir))
			#print(frames_list)
			sublist = []
			for idx in range(len(frames_list)):
				sublist.append(frames_list[idx])
				
				if idx < len(frames_list):
					frames = []
					for f in sublist:
						frame = Image.open(args.frames_dir + '/' + f)
						frame = tran(frame)
						frames.append(frame)
					frames = torch.stack(frames)
					#check
					#frames = frames[:-1] - frames[1:]
					#print(frames)
					print('classifying...')
					#print(frames_list[idx])
					#confidence, label = test(frames, model)
					label=predict(frames)
					print(label)
					print(classes[label])
					sublist = []


if __name__ == '__main__':
	main()
'''
