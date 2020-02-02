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
from PIL import Image
import torchvision
import os, random
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models


def predict(image_path, model, topk, gpu_usage):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    
    # TODO: Implement the code to predict the class from an image file
    cuda = torch.cuda.is_available()

    if cuda and gpu_usage:
        model.cuda()
        print('cuda')
    else:
        model.cpu()
        print('cpu')
    model.eval()

    # The image
    image = process_image(image_path)
    
    # tranfer to tensor
    image = torch.from_numpy(np.array([image])).float()
    if cuda == True and gpu_usage:
        image = Variable(image.cuda())
    else:
        image = Variable(image)
    output = model.forward(image)
    
    probs = torch.exp(output).data
    prob = torch.topk(probs, topk)[0].tolist()[0] 
    index = torch.topk(probs, topk)[1].tolist()[0] 
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    labels = []
    for i in range(5):
        labels.append(ind[index[i]])
    return prob, labels

# Implement a function for the validation pass
def validation(model, testloader, criterion, cuda):
    test_loss = 0
    accuracy = 0
    for data in testloader:
        images, labels = data
        if cuda == True:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

# A function that loads a checkpoint and rebuilds the model
def load_checkpoint(filename):
    
    checkpoint = torch.load(filename)
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
        
    return model, optimizer


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img = img.resize((256,256))
    img = img.crop((16,16,240,240))
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    return img.transpose(2,0,1)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


