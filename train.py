# Imports here
import numpy as np
import pandas as pd
import time
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import torchvision
from torchvision import datasets, transforms, models
import json
import helper


# Creates Argument Parser object named parser
parser = argparse.ArgumentParser()
    
# Argument 1: that's a path to a folder
parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'path to the folder of flower images') 
parser.add_argument('--arch', type = str, default = 'vgg', help = 'The CNN model architecture to use') 
parser.add_argument('--save_dir', type = str, default = '', help = 'path to the folder of saved model ')
parser.add_argument('--learning_rate', type = float, default = .001, help = 'learning rate value ')
parser.add_argument('--hidden_units', type = int, default = 4096, help = 'hidden_units number ')
parser.add_argument('--epochs', type = int, default = 11, help = 'number of epochs ')
parser.add_argument('--gpu', type = bool, default = True, help = 'training device ')
in_args = parser.parse_args()

data_dir = in_args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ])

cost_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ])

test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])
                                     ])


# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=cost_transforms)
test_data  = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the transforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
test_loader  = torch.utils.data.DataLoader(test_data, batch_size=32)

image_datasets = [train_data, valid_data, test_data]
dataloaders = [train_loader, valid_loader, test_loader]

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

arch = in_args.arch
arch = arch.lower()
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}
model = models[arch]

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, in_args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop1',nn.Dropout(.6)),
                          ('fc2', nn.Linear(in_args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr= in_args.learning_rate)

epochs = in_args.epochs
steps = 0
running_loss = 0
print_every = 103

cuda = torch.cuda.is_available()

if cuda and in_args.gpu:
    model.cuda()
    print('cuda')
    gpu_usage = True
else:
    model.cpu()
    print('cpu')
    gpu_usage = False
start = time.time()
for e in range(epochs):
    model.train()
    for data in dataloaders[0]:
        images, labels = data
        steps += 1
        
        
        optimizer.zero_grad()
        if gpu_usage == True:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = helper.validation(model, dataloaders[1], criterion, gpu_usage)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(dataloaders[1])),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders[1])))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()
total_time = time.time() - start
print("\nTotal time: {:.0f}m {:.0f}s".format(total_time//60, total_time % 60))


# Do validation on the test set
model.eval()
            
# Turn off gradients for validation, saves memory and computations
with torch.no_grad():
    test_loss, accuracy = helper.validation(model, dataloaders[2], criterion, gpu_usage)
                
    print("Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders[2])),
    "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders[2])))
    


# TODO: Save the checkpoint 
model.class_to_idx = image_datasets[0].class_to_idx

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': 'vgg16',
              'learning_rate': 0.001,
              'batch_size': 64,
              'classifier' : classifier,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

if(in_args.save_dir==''):
    torch.save(checkpoint, 'checkpoint.pth')
else:
    torch.save(checkpoint, in_args.save_dir + 'checkpoint.pth')