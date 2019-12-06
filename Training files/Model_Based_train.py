#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
from torch import nn
import torchvision.models as models
import torchvision
import os
import cv2
import sys
from sklearn.utils import shuffle
from Model_Based.py import predictor


# In[ ]:


model = predictor()

num_epochs = 50
batch_size = 1

data = np.load('finalX_10.npz')
data = data['arr_0']
label = np.load('finalY_10.npz')
label = label['arr_0']

data, label = shuffle(data, label)


criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    
    for it in range(data.shape[0]):    
        
        img = cv2.resize(data[it,:,:,:],(256,256))
        transform = torchvision.transforms.ToTensor()
        img = transform(img).unsqueeze(0)
        output = model.forward(img).squeeze()
    
        transform = torchvision.transforms.ToTensor()
        lab =torch.tensor(label[it]) 
        lab = lab[:,0]
        
        loss = criterion(output,lab.type(torch.FloatTensor))      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    print("Epoch:", epoch, "Loss: ", loss.item())

