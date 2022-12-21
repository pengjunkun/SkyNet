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
from weight_net_train import Net
import math
import argparse


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


app = FaceAnalysis(name='0')
app.prepare(ctx_id=0, det_size=(640, 640))

#SVM分类器
clf=joblib.load('userdata/clf.pkl')

#线程池
pool = ThreadPoolExecutor(max_workers=5)

#加载对比数据库
names = torch.load("./userdata/names.pt")#数据库的人名
embeddings = torch.load("./userdata/database.pt")#数据库每张图的特征向量
names_label=preprocessing.LabelEncoder().fit_transform(names)#将人名转化为数字标签
label2name=dict(zip(names_label,names))#创建标签与人名的字典
#font="./userdata/simhei.ttf"

save_dir='./userdata/fusion_weight.pth'
checkpoint=torch.load(save_dir)
fusion_net = Net()
fusion_net.load_state_dict(checkpoint['net'])

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

#单张图片人脸识别：输入一张图，用于测试
def detect_frame(img):
    image=np.array(img)
    image = image.astype(np.float32)
    faces = app.get(image)
    if faces:
        face=faces[0]
        face_embedding=face.embedding
        probs = [np.linalg.norm(face_embedding - embeddings[i]).item() for i in range(len(embeddings))]
        print(1/probs[0])



def draw_face(image,face_kps,bbox,i,objnum):
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

def x_pose(kps):
    dx1=kps[2][0]-(kps[0][0]+kps[3][0])/2
    dx2=kps[2][0]-(kps[1][0]+kps[4][0])/2
    if dx1*dx2<0:
        return True
    else:
        return False

def y_pose(kps):
    dy=kps[2][1]-(kps[3][1]+kps[4][1])/2
    if dy<0:
        return True
    else:
        return False

def level_allocation(pose):
    if pose[0]==True:
        if pose[1]==True:
            level=3
        else:
            level=2
    else:
        if pose[1]==True:
            level=1
        else:
            level=0
    return level

def weight_allocation(levels):
    piece=np.sum(levels)
    weight=np.array(levels/piece)
    return weight


#线程执行函数，输入图片，输出图片的特征向量
def detect(img,app,i,objnum):
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
        draw_face(image,face_kps,face_box,i,objnum)
        for j in range(face_kps.shape[0]):
            face_kps[j][0]=face_kps[j][0]/w-face_kps[0][0]/w
            face_kps[j][1]=face_kps[j][1]/h-face_kps[0][1]/h
        kps=face_kps.reshape((1,10))
        weight=fusion_net(torch.tensor(kps)).detach().numpy()
        return face_embedding,1,weight
    else:
        return 0,0,0


#多线程的多输入检测
def multi_detect_frame(images,objnum,fusion_mode):

    #存储线程
    thr=[]
    #开始计时
    s=time.time()
    #为每一张图创建一个线程
    for i,img in enumerate(images):
        thr_temp=pool.submit(detect, img,app,i,objnum)
        thr.append(thr_temp)
    #等待所有线程的完成
    res = wait([thr[i] for i in range(len(images))],return_when=ALL_COMPLETED)
    #获取线程执行的函数的返回结果
    faces_embedding=None
    for i,r in enumerate(res.done):
        result,flag,weight=r.result()
        if flag!=0:
            result=np.expand_dims(result,axis=0)
            #print(pose)
            diff = [np.subtract(result, embeddings[i]) for i in range(len(embeddings))]
            dist = [np.sum(np.square(diff[i]),1) for i in range(len(diff))]
            #name=names[np.argmin(dist)]
            print(np.min(dist))

            if faces_embedding is None:
                faces_embedding=result
                faces_weight=weight
            else:
                faces_embedding=np.concatenate((faces_embedding,result),axis=0)
                faces_weight=np.concatenate((faces_weight,weight),axis=0)

    #使用训练的SVM分类器对提取的向量进行分类
    if faces_embedding is None:
        name='unknow'
        min_dis='infty'
    else:
        if fusion_mode=='arcfusion':
            if faces_weight.shape[0]==1:
                final_face_embedding=faces_embedding
            else:
                #weight=np.expand_dims(weight_allocation(levels),axis=1)
                faces_weight=normalization(faces_weight)/np.sum(normalization(faces_weight))
                final_face_embedding=np.expand_dims(np.sum(faces_weight*faces_embedding,axis=0),axis=0)#将所有图片的向量进行均值
        else:
            final_face_embedding=np.expand_dims(np.mean(faces_embedding,axis=0),axis=0)#将所有图片的向量进行均值


        #注释掉的方法是计算待检测图片的向量到数据库其他向量的距离，显然随着数据库的增加，计算时间会增大，因此实现了其他方法
        '''
        final_face_embedding=torch.mean(faces_embedding,dim=0)
        # 计算距离
        probs = [(final_face_embedding - embeddings[i]).norm().item() for i in range(embeddings.size()[0])]
        # 我们可以认为距离最近的那个就是最有可能的人，但也有可能出问题，数据库中可以存放一个人的多视角多姿态数据，对比的时候可以采用其他方法，如投票机制决定最后的识别人脸
        index = probs.index(min(probs))   # 对应的索引就是判断的人脸
        name = names[index] # 对应的人脸
        '''
        #SVM
        #result_label=clf.predict(final_face_embedding)#使用SVM分类，得到label
        #name=label2name[result_label[0]]#提取label对应的结果

        #欧式距离
        diff = [np.subtract(final_face_embedding, embeddings[i]) for i in range(len(embeddings))]
        dist = [np.sum(np.square(diff[i]),1) for i in range(len(diff))]
        #设置距离阈值，以实现开集识别
        if np.min(dist)<400:
            name=names[np.argmin(dist)]
            min_dis=np.min(dist)
        else:
            name='unknow'
            min_dis=np.min(dist)
        #print('final:',name,np.min(dist))

        #余弦距离
        #similarity= [ np.sum(np.multiply(final_face_embedding, embeddings[i]), axis=1)/np.linalg.norm(final_face_embedding[0], axis=0) * np.linalg.norm(embeddings[i], axis=0) for i in range(len(embeddings))]
        #dist = [np.arccos(similarity[i]) / math.pi for i in range(len(embeddings))]

        #print(1)




    #计算时间消耗
    e=time.time()
    print("时间消耗：",e-s)
    return name,min_dis



if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("-t","--time",help="time stemp",default=1)
    ap.add_argument("-f","--fusion",help="fusion mode",default='arcfusion')
    args=vars(ap.parse_args())

    key = str(args["time"])
    fusion_mode = str(args["fusion"])

    #fusion_mode='arcfusion'
    #读取图片文件夹
    img_path='test/'+key+'/'
    object_names = os.listdir(img_path)

    #网络的热身，让整个网络检测更快
    warm_img=cv2.imread('warm_input.jpg')
    app.get(warm_img)

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

        #人脸检测+融合识别
        name,distance=multi_detect_frame(images,i+1,fusion_mode)
        IDs.append(name)
        object_dis.append(distance)

        print("Object",i+1,"ID:",name,'; distance:',distance)

    end=time.time()

    file=open('result.TXT','a')
    file.write('time stemp:'+key+'\n')
    for i,ID in enumerate(IDs):
        file.write(object_names[i]+':'+ID+'; distance:'+str(object_dis[i])+'\n')

    #关闭线程池
    pool.shutdown()
    file.close()

    print("总识别消耗:",end-start)


