import torch
import os
from PIL import Image,ImageFont,ImageDraw
import time
from concurrent.futures import ThreadPoolExecutor,wait,ALL_COMPLETED
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import joblib
from sklearn import preprocessing
import numpy as np
from insightface.app import FaceAnalysis
import cv2
from module.Identification.weight_net_train import Net
import math
import argparse


class Identifier(object):
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #识别器
        self.app = FaceAnalysis(name='10')
        self.app.prepare(ctx_id=0, det_size=(160, 160))
        warm_img=cv2.imread('module/Identification/warm_input.jpg')
        self.app.get(warm_img)
        #线程池
        self.pool = ThreadPoolExecutor(max_workers=5)
        #数据库
        self.names = torch.load("module/Identification/userdata/names.pt")#数据库的人名
        self.embeddings = torch.load("module/Identification/userdata/database.pt")#数据库每张图的特征向量
        self.names_label=preprocessing.LabelEncoder().fit_transform(self.names)#将人名转化为数字标签
        self.label2name=dict(zip(self.names_label,self.names))#创建标签与人名的字典
        #人脸角度网络
        checkpoint=torch.load('module/Identification/userdata/fusion_weight.pth')
        self.fusion_net = Net()
        self.fusion_net.load_state_dict(checkpoint['net'])


    def normalization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def draw_face(self,image,face_kps,bbox,i,objnum):
        for j in range(face_kps.shape[0]):
            color = (0, 0, 255)
            if j == 0 or j == 3:
                color = (0, 255, 0)
            x=int(face_kps[j][0])
            y=int(face_kps[j][1])
            #cv2.circle(image,(x,y), 2, color ,2)

        shape=image.shape
        w=shape[0]
        h=shape[1]
        x1=max(int(bbox[1]),0)
        x2=min(int(bbox[3]),w)
        y1=max(int(bbox[0]),0)
        y2=min(int(bbox[2]),h)
        center=[(x1+x2)/2,(y1+y2)/2]
        size=max(x2-x1,y2-y1)
        x1=max(int(center[0]-size/2),0)
        x2=min(int(center[0]+size/2),w)
        y1=max(int(center[1]-size/2),0)
        y2=min(int(center[1]+size/2),h)
        face_img=image[x1:x2,y1:y2,:]
        #cv2.putText(face_img, '('+str(pose[0])+','+str(pose[1])+')', (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        face_img=cv2.resize(face_img,(160,160))
        face_img=cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        path='test_result/object'+str(objnum)
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path+'/'+'crop_'+str(i)+'.jpg',face_img)

    #线程执行函数，输入图片，输出图片的特征向量
    def detect(self,img,app,i,objnum):
        image=np.array(img)
        image = image.astype(np.float32)

        faces = app.get(image)
        shape=image.shape
        h=shape[0]
        w=shape[1]

        if faces:
            face=faces[0]
            face_embedding=face.embedding
            face_kps=face.kps
            face_box=face.bbox
            #pose=(x_pose(face_kps),y_pose(face_kps))
            #level=level_allocation(pose)
            #self.draw_face(image,face_kps,face_box,i,objnum)
            for j in range(face_kps.shape[0]):
                face_kps[j][0]=face_kps[j][0]/w-face_kps[0][0]/w
                face_kps[j][1]=face_kps[j][1]/h-face_kps[0][1]/h
            kps=face_kps.reshape((1,10))
            weight=self.fusion_net(torch.tensor(kps)).detach().numpy()
            return face_embedding,1,weight
        else:
            return 0,0,0

    #多线程的多输入检测
    def multi_detect_frame(self,images,objnum,fusion_mode):

        #存储线程
        thr=[]
        #开始计时
        #s=time.time()
        #为每一张图创建一个线程
        for i,img in enumerate(images):
            if img is None:
                continue
            thr_temp=self.pool.submit(self.detect, img ,self.app,i,objnum)
            thr.append(thr_temp)
        #等待所有线程的完成
        res = wait([thr[i] for i in range(len(thr))],return_when=ALL_COMPLETED)
        #获取线程执行的函数的返回结果
        faces_embedding=None
        for i,r in enumerate(res.done):
            result,flag,weight=r.result()
            if flag!=0:
                result=np.expand_dims(result,axis=0)
                #print(pose)
                #diff = [np.subtract(result, self.embeddings[i]) for i in range(len(self.embeddings))]
                #dist = [np.sum(np.square(diff[i]),1) for i in range(len(diff))]
                #name=names[np.argmin(dist)]
                dist=[np.linalg.norm(result - self.embeddings[i]).item() for i in range(len(self.embeddings))]

                #print(np.min(dist))

                if faces_embedding is None:
                    faces_embedding=result
                    faces_weight=weight
                else:
                    faces_embedding=np.concatenate((faces_embedding,result),axis=0)
                    faces_weight=np.concatenate((faces_weight,weight),axis=0)

        #使用训练的SVM分类器对提取的向量进行分类
        if faces_embedding is None:
            name='others'
            min_dis='infty'
        else:
            if fusion_mode=='arcfusion':
                if faces_weight.shape[0]==1:
                    final_face_embedding=faces_embedding
                else:
                    #weight=np.expand_dims(weight_allocation(levels),axis=1)
                    faces_weight=self.normalization(faces_weight)/np.sum(self.normalization(faces_weight))
                    final_face_embedding=np.expand_dims(np.sum(faces_weight*faces_embedding,axis=0),axis=0)#将所有图片的向量进行均值
            else:
                final_face_embedding=np.expand_dims(np.mean(faces_embedding,axis=0),axis=0)#将所有图片的向量进行均值

            #欧式距离
            #diff = [np.subtract(final_face_embedding, self.embeddings[i]) for i in range(len(self.embeddings))]
            #dist = [np.sum(np.square(diff[i]),1) for i in range(len(diff))]
            dist=[np.linalg.norm(final_face_embedding - self.embeddings[i]).item() for i in range(len(self.embeddings))]

            #设置距离阈值，以实现开集识别
            if np.min(dist)<20:
                name=self.names[np.argmin(dist)]
                min_dis=np.min(dist)
            else:
                name='others'
                min_dis=np.min(dist)

        #计算时间消耗
        #e=time.time()
        #print("时间消耗：",e-s)
        return name,min_dis

if __name__ == '__main__':
    identifer=Identifier()
    img_path='test/'+str(1)+'/'
    object_names = os.listdir(img_path)
    #遍历每一个物体的图片，进行人脸检测与人脸识别
    IDs=[]
    object_dis=[]
    start=time.time()
    for i,object_name in enumerate(object_names):
        img_names = os.listdir(img_path+object_name+'/')
        images=[]
        for img_name in img_names:
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                #存储该物体的图片
                #print(object_name,img_name)
                #detect_frame(Image.open(img_path+object_name+'/'+img_name))
                images.append(Image.open(img_path+object_name+'/'+img_name))

        name,distance=identifer.multi_detect_frame(images,i+1,'arcfusion')
        print(name,distance)
