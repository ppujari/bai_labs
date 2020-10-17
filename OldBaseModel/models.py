import torch
import argparse
#from smoker_detector_base import in_args
from torchvision import datasets, transforms, utils
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler
from collections import OrderedDict

def alexnet(class_names):
	print("In alexnet")
	model = models.alexnet(pretrained = True)

	classifier = nn.Sequential(OrderedDict([
	                          ('fc1', nn.Linear(9216, 4096)),
	                          ('relu', nn.ReLU()),
	                          ('fc2', nn.Linear(4096, len(class_names))),
	                          ('output', nn.LogSoftmax(dim=1))
	                          ]))

	for param in model.parameters():
	    param.requires_grad = False

	model.classifier = classifier

	if torch.cuda.is_available():
	    model.cuda()
	return model

def resnet18(class_names):
	print("IN resnet18")
	model=models.resnet18(pretrained=True)

	'''
	This is for resnet 50
	classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 512)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(0.2)),
                          ('fc2', nn.Linear(512,len(class_names))),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
	'''

	fc= nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(512, 100)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(100,len(class_names))),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
	#num_in_features=model.fc.in_features
	#cls_num=len(class_names)
	#model.fc.out_features=cls_num
	model.fc=fc

	#for param in model.parameters():
	#   param.requires_grad = False
	# if the above loop isnt included, required_grad=True by default. 
	#this is required when we pass resnet as the model

	#model.classifier = classifier

	if torch.cuda.is_available():
	    model.cuda()
	return model


def vgg19(class_names):
	print("IN vgg19")
	model=models.vgg19(pretrained=True)

	classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096,len(class_names))),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

	for param in model.parameters():
	    param.requires_grad = False

	model.classifier = classifier

	if torch.cuda.is_available():
	    model.cuda()

	return model

def getModel(class_names,model_name):
	model=globals()[model_name](class_names)
	return model
