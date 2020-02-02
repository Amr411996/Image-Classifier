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
import os, random
import matplotlib.pyplot as plt

# Creates Argument Parser object named parser
parser = argparse.ArgumentParser()
    
# Argument 1: that's a path to a folder
parser.add_argument('--img_path', type = str, default = 'flowers/test/10/image_07090.jpg', help = 'path to the image')  
parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'path to the saved model ')
parser.add_argument('--top_k', type = int, default = 5, help = 'top classes number ')
parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'path to real category names ')
parser.add_argument('--gpu', type = bool, default = True, help = 'training device ')
in_args = parser.parse_args()

with open(in_args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
model, optimizer = helper.load_checkpoint(in_args.checkpoint)


img_path = in_args.img_path
prob, classes = helper.predict(img_path, model, in_args.top_k, in_args.gpu)
max_index = np.argmax(prob)
max_probability = prob[max_index]
label = classes[max_index]

#fig = plt.figure(figsize=(8,8))
#ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
#ax2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)

#image = Image.open(img_path)
#ax1.axis('off')
#ax1.set_title(cat_to_name[label])
#ax1.imshow(image)

labels = []
for cl in classes:
    labels.append(cat_to_name[cl])
    
#y_pos = np.arange(5)
#ax2.set_yticks(y_pos)
#ax2.set_yticklabels(labels)
#ax2.set_xlabel('Probability')
#ax2.invert_yaxis()
#ax2.barh(y_pos, prob, xerr=0, align='center', color='green')

#plt.show()

print('flower name: ', cat_to_name[label])
print('classes probabilities')
for i in range(len(labels)):
    print(labels[i],' : ', prob[i])