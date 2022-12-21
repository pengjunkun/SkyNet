# 制作人脸特征向量的数据库 最后会保存两个文件，分别是数据库中的人脸特征向量和对应的名字。当然也可以保存在一起
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
import cv2
from tqdm import tqdm


#standard_names = torch.load("./userdata/standard_names.pt")#数据库的人名
#standard_embeddings = torch.load("./userdata/standard_database.pt")#数据库每张图的特征向量
#standard=dict(zip(standard_names,standard_embeddings))

app = FaceAnalysis(name='10')
app.prepare(ctx_id=0, det_size=(640,640))

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))



def collate_fn(x):
    return x[0]
# 将所有的单人照图片放在各自的文件夹中，文件夹名字就是人的名字,存放格式如下
'''
--orgin
  |--zhangsan
     |--1.jpg
     |--2.jpg
  |--lisi
     |--1.jpg
     |--2.jpg
'''
dataset = datasets.ImageFolder('./img')  #加载数据库
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
aligned = []  # aligned就是从图像上抠出的人脸，大小是之前定义的image_size=160
names = []
embeddings=[]
kps=[]
weight=[]
i= 1
for x, y in tqdm(loader):
    path = './crop_img/{}/'.format(dataset.idx_to_class[y])  # 这个是要保存的人脸路径
    if not os.path.exists(path):
        i = 1
        os.mkdir(path)
    # 如果要保存识别到的人脸，在save_path参数指明保存路径即可,不保存可以用None

        #print('Face detected with probability: {:8f}'.format(prob))
    x_data=np.array(x)
    x_data = x_data.astype(np.float32)
    faces=app.get(x_data)

    if faces:
        x_data = cv2.cvtColor(x_data, cv2.COLOR_BGR2RGB)
        #shape=x_data.shape
        #h=shape[0]
        #w=shape[1]
        rimg = app.draw_on(x_data, [faces[0]])
        cv2.imwrite(path+str(i)+'.jpg', rimg)
        i+=1
        face = faces[0]
        #box=face.bbox
        #kps_now=face.kps
        #for j in range(kps_now.shape[0]):
            #kps_now[j][0]=kps_now[j][0]/w-kps_now[0][0]/w
            #kps_now[j][1]=kps_now[j][1]/h-kps_now[0][1]/h

        #probs = np.linalg.norm(face.embedding - standard[dataset.idx_to_class[y]]).item()
        #if probs<1:
            #print(dataset.idx_to_class[y],1)
            #continue
        #kps_now=kps_now.reshape((1,10))
        #kps.append(kps_now)
        names.append(dataset.idx_to_class[y])
        embeddings.append(face.embedding)
        #weight.append(1/probs)
    else:
        print(dataset.idx_to_class[y],2)




#aligned = torch.stack(aligned).to(device)
#embeddings = resnet(aligned).detach().cpu()   # 提取所有人脸的特征向量，每个向量的长度是512
# 两两之间计算混淆矩阵



#dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
#print(names)
#print(pd.DataFrame(dists, columns=names, index=names))
torch.save(embeddings,'./userdata/database.pt')  # 当然也可以保存在一个文件
torch.save(names,'./userdata/names.pt')
#torch.save(kps,'./userdata/kps.pt')
#torch.save(weight,'./userdata/weight.pt')
