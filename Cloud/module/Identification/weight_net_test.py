import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from weight_net_train import Net
from insightface.app import FaceAnalysis
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

save_dir='./userdata/fusion_weight.pth'

checkpoint=torch.load(save_dir)
net = Net()
net.load_state_dict(checkpoint['net'])
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(160, 160))


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

img_path='test/object1/'
imgnames=os.listdir(img_path)
for i,imgname in enumerate(imgnames):
    img=cv2.imread(img_path+imgname)
    shape=img.shape
    h=shape[0]
    w=shape[1]
    faces=app.get(img)
    if faces:
        box=faces[0].bbox
        kps=faces[0].kps
        for j in range(kps.shape[0]):
            kps[j][0]=kps[j][0]/w-kps[0][0]/w
            kps[j][1]=kps[j][1]/h-kps[0][1]/h
        kps=kps.reshape((1,10))
        weight=net(torch.tensor(kps)).detach().numpy()
        if i==0:
            img_weight=weight
        else:
            img_weight=np.concatenate((img_weight,weight),axis=0)

w=normalization(img_weight)/np.sum(normalization(img_weight))
for i in range(w.shape[0]):
    print('object',i+1,':',w[i])
print(img_weight/np.sum(img_weight))

print(np.sum(normalization(img_weight)))
print(np.sum(img_weight))


