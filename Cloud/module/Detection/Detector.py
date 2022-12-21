import torch
from PIL import ImageDraw, ImageFont
import math
import time
import cv2
import numpy as np
import argparse
from PIL import Image
from module.Detection.yolo import YOLO
import module.Detection.yolo_barricade as yolo_barricade
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor,wait,ALL_COMPLETED
from insightface.app import FaceAnalysis

class Detector(object):

    def __init__(self,drone_id):
        #定义无人机的基本信息
        self.drone_id=drone_id
        #定义检测网络
        self.warm_input=Image.open('module/Detection/warm_up.jpg')
        self.yolo = YOLO()
        self.yolo_b = yolo_barricade.YOLO()
        self.yolo_b.detect_image(self.warm_input)
        self.yolo.detect_image(self.warm_input)
        #定义线程池
        self.pool=ThreadPoolExecutor(max_workers=5)
        #定义人脸检测网络
        self.app=FaceAnalysis(name='0')
        self.app.prepare(ctx_id=10, det_size=(640, 640))
        self.app.get(np.array(self.warm_input)[...,::-1])

    def detect(self,yolo_now,image):
        r_image,boxes,conf,classnames = yolo_now.detect_image(image)
        return r_image,boxes,conf,classnames

    def predict(self,img_path,result_path,time_stemp):
        s=time.time()

        #读取文件
        image=Image.open(img_path)
        image_data=np.array(image)[...,::-1]

        #提交检测任务到线程池
        thr1=self.pool.submit(self.detect,self.yolo,image)
        thr2=self.pool.submit(self.detect,self.yolo_b,image)

        #获取检测结果
        r_image,boxes,conf,classnames=thr1.result()
        r_image2,boxes2,conf2,classnames2=thr2.result()
        e=time.time()
        print("检测消耗：",e-s)

        #保存结果
        sub_save_path=result_path+'bb_'+str(self.drone_id)+'_'+str(time_stemp)+'/'
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)
        txt_file_path = sub_save_path +  'box.txt'
        txt_file = open(txt_file_path,'w')
        print(image.size[0],image.size[1],file=txt_file)

        #人脸检测
        index=1
        for i,box in enumerate(boxes) :
            #获取人物图像
            object_img=image_data[box[0]:box[2],box[1]:box[3],:]
            face_shape=object_img.shape
            #对人物图像进行人脸检测
            face_bboxes, kpss = self.app.det_model.detect(object_img,max_num=0,metric='default')
            if not (face_bboxes.shape[0] ==0):
                #提取人脸图片
                face_bbox = face_bboxes[0, 0:4]
                w=max(face_bbox[2]-face_bbox[0],face_bbox[3]-face_bbox[1])
                center=[(face_bbox[2]+face_bbox[0])/2,(face_bbox[3]+face_bbox[1])/2]
                face_bbox[0]=max(0,center[0]-w/2)
                face_bbox[1]=max(0,center[1]-w/2)
                face_bbox[2]=min(face_shape[1],center[0]+w/2)
                face_bbox[3]=min(face_shape[0],center[1]+w/2)
                face_bbox=face_bbox.astype('int')
                face_img=object_img[face_bbox[1]:face_bbox[3],face_bbox[0]:face_bbox[2],:]
                #保存人脸图片
                cv2.imwrite(sub_save_path+str(self.drone_id)+'_'+str(index)+'.jpg',face_img)
            #保存目标人物的信息
            box_id= str(self.drone_id)+'_'+str(index)
            print(box_id,':',box[0],box[1],box[2],box[3],file=txt_file)
            index+=1
        txt_file.close()

        #保存特征点的信息
        txt_file_path = sub_save_path +  'featurepoint.txt'
        txt_file = open(txt_file_path,'w')
        dic_point=dict()
        for i,box in enumerate(boxes2) :
            #用box的中心点代表特征点的二维坐标
            center=(int((box[1]+box[3])/2),int((box[0]+box[2])/2))
            label= classnames2[i][-1:]
            dic_point[label]=center
        for i in range(len(boxes2)):
            print(dic_point[str(i)][0],dic_point[str(i)][1],file=txt_file)
        txt_file.close()

    def predict_single(self,img_path,result_path,time_stemp):
        s=time.time()

        #读取文件
        image=Image.open(img_path)
        image_data=np.array(image)[...,::-1]

        #获取检测结果
        r_image,boxes,conf,classnames = self.yolo.detect_image(image)

        e=time.time()
        print("检测消耗：",e-s)

        #保存结果
        sub_save_path=result_path+'bb_'+str(self.drone_id)+'_'+str(time_stemp)+'/'
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)
        txt_file_path = sub_save_path +  'box.txt'
        txt_file = open(txt_file_path,'w')
        print(image.size[0],image.size[1],file=txt_file)

        #人脸检测
        index=1
        for i,box in enumerate(boxes) :
            #获取人物图像
            object_img=image_data[box[0]:box[2],box[1]:box[3],:]
            face_shape=object_img.shape
            #对人物图像进行人脸检测
            face_bboxes, kpss = self.app.det_model.detect(object_img,max_num=0,metric='default')
            if not (face_bboxes.shape[0] ==0):
                #提取人脸图片
                face_bbox = face_bboxes[0, 0:4]
                w=max(face_bbox[2]-face_bbox[0],face_bbox[3]-face_bbox[1])
                center=[(face_bbox[2]+face_bbox[0])/2,(face_bbox[3]+face_bbox[1])/2]
                face_bbox[0]=max(0,center[0]-w/2)
                face_bbox[1]=max(0,center[1]-w/2)
                face_bbox[2]=min(face_shape[1],center[0]+w/2)
                face_bbox[3]=min(face_shape[0],center[1]+w/2)
                face_bbox=face_bbox.astype('int')
                face_img=object_img[face_bbox[1]:face_bbox[3],face_bbox[0]:face_bbox[2],:]
            #保存人脸图片


            cv2.imwrite(sub_save_path+str(self.drone_id)+'_'+str(index)+'.jpg',face_img)
            #保存目标人物的信息
            box_id= str(self.drone_id)+'_'+str(index)
            print(box_id,':',box[0],box[1],box[2],box[3],file=txt_file)
            index+=1
        txt_file.close()

if __name__ == "__main__":
    drone1=Detector(1)
    drone1.predict('img/1_1.jpg','img_out/',1)




