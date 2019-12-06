import numpy as np
import torch
from torch import nn
import torchvision.models as models
import torchvision
import os
import cv2
import sys
from sklearn.utils import shuffle


deformation_matrix = np.load('deformation_matrix.npz')['arr_0']
base = np.load('base_vertices.npz')['arr_0']


class predictor(torch.nn.Module):

    def __init__(self):
        super(predictor, self).__init__()

        self.model = models.resnet18(pretrained = True)
        self.model.fc = nn.Linear(512, 256)

        self.encoder = nn.Sequential(
            self.model,
            nn.ReLU(inplace=True),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,15)
            )


    def forward(self,x):
        output = self.encoder(x.float())
        return output


model = predictor()
weights = torch.load("paramteric.pth")
model.apply(weights)
data = cv2.imread(path)
input_img = cv2.resize(data,(256,256))
transform = torchvision.transforms.ToTensor()
img = transform(img).unsqueeze(0)
output = model(input_img)



new_def_2 = np.matmul(deformation_matrix,output)
new_model_2 = base + new_def_2
new_model_2 = new_model_2.reshape(300,3)
#new_model_2 is sent to matlab for visualizing .obj file
a={}
a['f'] = new_model_2
savemat('coeff.mat',a)

