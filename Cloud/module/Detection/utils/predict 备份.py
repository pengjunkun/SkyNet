#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#

from crop_utils import self_adaptive_crop
#from edsr import EDSR

import torch
from PIL import ImageDraw, ImageFont
import math

import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO

#from collections import deque
#from keras import backend
#import tensorflow as tf
#from tensorflow.compat.v1 import InteractiveSession
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)


#绕轴旋转
def RotateByZ(Cx, Cy, thetaZ):
    rz = thetaZ*math.pi/180.0
    outX = math.cos(rz)*Cx - math.sin(rz)*Cy
    outY = math.sin(rz)*Cx + math.cos(rz)*Cy
    return outX, outY
def RotateByY(Cx, Cz, thetaY):
    ry = thetaY*math.pi/180.0
    outZ = math.cos(ry)*Cz - math.sin(ry)*Cx
    outX = math.sin(ry)*Cz + math.cos(ry)*Cx
    return outX, outZ
def RotateByX(Cy, Cz, thetaX):
    rx = thetaX*math.pi/180.0
    outY = math.cos(rx)*Cy - math.sin(rx)*Cz
    outZ = math.sin(rx)*Cy + math.cos(rx)*Cz
    return outY, outZ

def get_camera_position(object_3d_points,object_2d_point1,camera_matrix,dist_coefs):
# 求解相机位姿
    found1, rvec1, tvec1 = cv2.solvePnP(object_3d_points, object_2d_point1, camera_matrix, dist_coefs)
    rotM1 = cv2.Rodrigues(rvec1)[0]

    #camera_postion1 = -np.matrix(rotM1).T * np.matrix(tvec1)

# 计算相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。旋转顺序z,y,x
    thetaZ1 = math.atan2(rotM1[1, 0], rotM1[0, 0])*180.0/math.pi
    thetaY1 = math.atan2(-1.0*rotM1[2, 0], math.sqrt(rotM1[2, 1]**2 + rotM1[2, 2]**2))*180.0/math.pi
    thetaX1 = math.atan2(rotM1[2, 1], rotM1[2, 2])*180.0/math.pi
# 相机坐标系下值
    x1 = tvec1[0]
    y1 = tvec1[1]
    z1 = tvec1[2]

# 进行三次旋转
    (x1, y1) = RotateByZ(x1, y1, -1*thetaZ1)
    (x1, z1) = RotateByY(x1, z1, -1*thetaY1)
    (y1, z1) = RotateByX(y1, z1, -1*thetaX1)
#获得相机坐标系在世界坐标系的位置坐标
    Cx1 = x1*-1
    Cy1 = y1*-1
    Cz1 = z1*-1

    p_camera=[Cx1,Cy1,Cz1]

    return p_camera,thetaX1,thetaY1,thetaZ1,rotM1,tvec1

#将图像坐标系转化到相机坐标系
def get_p_position(p,p_camera,camera_matrix,zc,thetaX,thetaY,thetaZ):
    fx=camera_matrix[0,0]
    u0=camera_matrix[0,2]
    fy=camera_matrix[1,1]
    v0=camera_matrix[1,2]

    xc=(p[0]-u0)*zc/fx;
    yc=(p[1]-v0)*zc/fy;
    pc=[xc,yc,zc]

    (xc, yc) = RotateByZ(xc, yc, -1*thetaZ)
    (xc, zc) = RotateByY(xc, zc, -1*thetaY)
    (yc, zc) = RotateByX(yc, zc, -1*thetaX)

    pw=[xc+p_camera[0][0],yc+p_camera[1][0],zc+p_camera[2][0]]

    return pw,pc

#三维点投影到二维图像点
def T2D(p,camera_matrix,rotM,tvec):
    Out_matrix = np.concatenate((rotM, tvec), axis=1)
    pixel = np.dot(camera_matrix, Out_matrix)
    pixel1 = np.dot(pixel,p)
    pixel2 = pixel1/pixel1[2]
    pixel2 = pixel2.astype(np.int)
    #print(pixel2)
    return pixel2

def test_view(view,p,camera_matrix,rotM,tvec):
    Out_matrix = np.concatenate((rotM, tvec), axis=1)
    pixel = np.dot(camera_matrix, Out_matrix)
    pixel1 = np.dot(pixel,p)
    pixel2 = pixel1/pixel1[2]
    pixel2 = pixel2.astype(np.int)
    print(pixel2)
    pixel_list=list(pixel2)

    img = cv2.imread('./img/'+view)
    result=img
    result[pixel_list[1]-10:pixel_list[1]+10,pixel_list[0]-10:pixel_list[0]+10,0]=0
    result[pixel_list[1]-10:pixel_list[1]+10,pixel_list[0]-10:pixel_list[0]+10,1]=255
    result[pixel_list[1]-10:pixel_list[1]+10,pixel_list[0]-10:pixel_list[0]+10,2]=0
    cv2.namedWindow("result",cv2.WINDOW_NORMAL)
    cv2.imshow("result",result)
    cv2.waitKey(0)

    view_name=view.split('.')[0]

    cv2.imwrite('./testimg/'+view_name+'_result.jpg',result)
    return pixel1[2],pixel2

def view_point(view,p,camera_matrix,rotM,tvec,box_num):
    Out_matrix = np.concatenate((rotM, tvec), axis=1)
    pixel = np.dot(camera_matrix, Out_matrix)
    img = cv2.imread('./img/'+view)
    result=img
    for i in range(box_num):
        if p[i]:
            for p_now in p[i]:
                pixel1 = np.dot(pixel,p_now)
                pixel2 = pixel1/pixel1[2]
                pixel2 = pixel2.astype(np.int)
                pixel_list=list(pixel2)

                result[pixel_list[1],pixel_list[0],0]=0
                result[pixel_list[1],pixel_list[0],1]=255
                result[pixel_list[1],pixel_list[0],2]=0

    cv2.namedWindow("result",cv2.WINDOW_NORMAL)
    cv2.imshow("result",result)
    cv2.waitKey(0)

    view_name=view.split('.')[0]

    cv2.imwrite('./testimg/'+view_name+'_result.jpg',result)
    return pixel1[2],pixel2

#在图像上绘制预测框
def draw_retangle(image,boxes):
    for box in boxes:
        top, left, bottom, right = box
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean([640,640]), 1))

        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))
        bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
        right   = min(image.size[0], np.floor(right).astype('int32'))


        draw = ImageDraw.Draw(image)
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i])

    del draw

    return image

#判断当前点是否在框内
def inbox(p,box):
    if p[1]>=box[0] and p[1]<=box[2] and p[0]>=box[1] and p[0]<=box[3]:
        return 1
    else:
        return 0

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-m","--mode",help="work mode")
    args=vars(ap.parse_args())


    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #mode =args["mode"]
    print(mode)
    #----------------------------------------------------------------------------------------------------------#
    UAV_id=1
    #   UAV_id用于指定无人机的编号
    #----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = "video_result.mp4"
    video_fps       = 25.0
    #-------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    #-------------------------------------------------------------------------#
    test_interval   = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #   dir_redetect_path为超分辨率后的图片保存路径
    #   dir_3D_path为对齐后按组分好的输出的图片路径
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    dir_redetect_path = "img/srimg"
    dir_3D_path = "img_out/get3D"

    #------------------------------------------------------------------------


    if mode == "predict":

        #1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。
        #2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        #3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        #在原图上利用矩阵的方式进行截取。
        #4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        #比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image,boxes,conf,classnames = yolo.detect_image(image)
                #r_image.show()
                r_image.save("result.jpg")

    elif mode == "video":#视频检测模式
        import os
        #video_path = input('Input video filename:')

        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        img_index=0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref,frame=capture.read()
            # 格式转变，BGRtoRGB
            if frame is None:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            row_frame=frame.copy()
            # 进行检测
            frame,boxes,conf,classnames=yolo.detect_image2(frame)

            #保存图片与box
            now_time=time.time()
            sub_save_path=dir_save_path+'bb_'+str(UAV_id)+'/'+'bb_'+str(UAV_id)+'_'+str(now_time)+'/'
            img_name=str(img_index)+'.jpg'
            img_index+=1

            os.makedirs(sub_save_path)
            frame.save(os.path.join(sub_save_path, img_name))

            txt_file_path = sub_save_path +'box.txt'
            txt_file = open(txt_file_path,'w')
            index=1
            for box in boxes:
                box_id= str(UAV_id)+'_'+str(index)
                print(box_id,':',box[0],box[1],box[2],box[3],file=txt_file)
                index+=1
            txt_file.close()

            frame = np.array(frame)
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            #time.sleep(0.1)
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        capture.release()
        out.release()
        cv2.destroyAllWindows()


    elif mode == "fps":#模型速度测试
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":#批量图片检测,模拟多个无人机
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        now_time=time.time()
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                image_data=np.array(image)[...,::-1]
                r_image,boxes,conf,classnames= yolo.detect_image(image)
                #r_image.show()
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)

                sub_save_path=dir_save_path+'align_bb_0/'+'bb_'+str(UAV_id)+'/'+'bb_'+str(UAV_id)+'_'+str(now_time)+'/'

                os.makedirs(sub_save_path)
                #image.save(os.path.join(sub_save_path, img_name))
                cv2.imwrite(os.path.join(sub_save_path, img_name),image_data)

                txt_file_path = sub_save_path +  'box.txt'
                txt_file = open(txt_file_path,'w')
                index=1
                for box in boxes:
                    box_id= str(UAV_id)+'_'+str(index)
                    print(box_id,':',box[0],box[1],box[2],box[3],file=txt_file)
                    index+=1
                txt_file.close()
                UAV_id+=1

    elif mode == "dir_predict_alignment":#对齐算法
        #超分辨率模型，图片识别可以用
        #sr_model = EDSR().to('cuda' if torch.cuda.is_available() else 'cpu')
        #sr_model.load_state_dict(torch.load('./model_data/visdrone_x2.pt'))

        import os

        from tqdm import tqdm

        #读取文件中的2d点集合
        with open("2Dpoint.txt", "r") as f:
            point2D = f.readlines()
            for i in range(len(point2D)):
                point2D[i]=point2D[i].strip('\n')

        #读取文件夹中的3d点集合
        with open("3Dpoint.txt", "r") as f:
            point3D_list = f.readlines()
            for i in range(len(point3D_list)):
                point3D_list[i]=point3D_list[i].strip('\n')

        #解析3d点
        point_num=len(point3D_list)
        for i in range(point_num):
            p3D=point3D_list[i].split(',')
            for j in range(len(p3D)):
                p3D[j]=int(p3D[j])
            point3D_list[i]=p3D

        #解析2d点
        point2D_list=[]
        for i in range(0,len(point2D),point_num+1):
            point2D_temp=[]
            for j in range(point_num):
                p2D=point2D[i+1+j].split(',')
                for k in range(len(p2D)):
                    p2D[k]=int(p2D[k])
                point2D_temp.append(p2D)
            point2D_list.append(point2D_temp)

        #相机的内参矩阵
        camera_matrix = np.array(([2659.56041592278	, 0, 2003.43050846023],
                        [0, 2852.73853758323, 1632.50749450792],
                        [0, 0, 1.0]), dtype=np.double)
        #相机的畸变参数
        dist_coefs = np.array([-0.0430766978645584, 0.0686908936174095, 0, 0, 0], dtype=np.double)

        #特征点世界坐标
        object_3d_points = np.array((point3D_list), dtype=np.double)

        #特征点在图的位置
        object_2d_point=[]
        for ptemp in point2D_list:
            object_2d_point_temp = np.array((ptemp), dtype=np.double)
            object_2d_point.append(object_2d_point_temp)


        imgnum=0
        boxes_allview=[]
        image_allview=[]
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):#遍历图片
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)

                #图像粗检测
                time_start=time.time()
                r_image,boxes,conf,classnames= yolo.detect_image(image)
                time_end=time.time()
                print('检测消耗：',time_end-time_start)

                #将得到的boundingbox与对应的视图添加到总集合中
                boxes_allview.append(list(boxes))
                print(boxes)
                print(conf)
                #r_image.show()
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
                image_cv2=cv2.imread(image_path, cv2.IMREAD_COLOR)
                image_allview.append(image_cv2)

                #自适应切割子区域
                time_start=time.time()
                corp_offsets, corp_imgs,crop_boxes=self_adaptive_crop(boxes, image_cv2)
                time_end=time.time()
                print('切分消耗：',time_end-time_start)
                image       = Image.open(image_path)
                crop_on_image=draw_retangle(image,crop_boxes)
                crop_on_image.save(os.path.join(dir_save_path, 'crop_'+img_name))
                count=1

                #超分
                sr_images=[]
                sr_num=1
                for corp_img in corp_imgs:
                    cv2.imwrite(dir_save_path+'/crop/'+str(sr_num)+'_crop_'+img_name,corp_img)
                    sr_name='sr_'+str(sr_num)+'_'+img_name
                    cv2.imwrite(dir_redetect_path+'/'+sr_name,corp_img)

                    '''
                    #超分辨率
                    sr_input = torch.FloatTensor(corp_img.copy()).to('cuda' if torch.cuda.is_available() else 'cpu').permute(2, 0, 1).unsqueeze(0)
                    image_data = sr_model(sr_input)
                    image_data = image_data.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
                    sr_images.append(image_data)
                    sr_name='sr_'+str(sr_num)+'_'+img_name
                    cv2.imwrite(dir_redetect_path+'/'+sr_name,image_data)

                    sr_num+=1
                
                    #重检测
                    image_path  = os.path.join(dir_redetect_path, sr_name)
                    image       = Image.open(image_path)
                    r_image,boxes,conf,classnames= yolo.detect_image(image)
                    #r_image.show()
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    r_image.save(os.path.join(dir_save_path, sr_name))
                    '''
                imgnum+=1

        #对齐算法
        p_camera=[]
        thetaX=[]
        thetaY=[]
        thetaZ=[]
        rotM=[]
        tvec=[]

        #计算每一副视图的旋转变换矩阵
        for i in range(imgnum):
            p_camera1,thetaX1,thetaY1,thetaZ1,rotM1,tvec1=get_camera_position(object_3d_points,object_2d_point[i],camera_matrix,dist_coefs)
            p_camera.append(p_camera1)
            thetaX.append(thetaX1)
            thetaY.append(thetaY1)
            thetaZ.append(thetaZ1)
            rotM.append(rotM1)
            tvec.append(tvec1)

        #开始对齐
        time_start=time.time()
        object_img=[]#所有对象的图片集合
        boxes_index=[]#所有对象的boundingbox集合
        #三维点空间的遍历
        for i in range(-1000,1000,10):
            for j in range(-1000,1000,10):
                for k in range(150,250,10):
                    p=[i,j,k,1]#当前三维点
                    object_img_temp=[]#当前三维点对应的对象的图片集合
                    boxes_index_temp=set()#当前三维点对应的对象的boundingbox集合
                    #将三维点投影到每一副视图上，检测是否在框中
                    for view in range(imgnum):
                        #获得三维点在二维图像的投影位置
                        piexl_now=T2D(np.array(p,dtype=np.double),camera_matrix,rotM[view],tvec[view])
                        index=0
                        #判断是否在框中
                        for box in boxes_allview[view]:
                            if inbox(piexl_now,box):
                                #在框中则将其添加到当前三维点对应的图片集合与boundingbox集合中
                                object_img_temp.append(image_allview[view][box[0]:box[2],box[1]:box[3],:])
                                boxes_index_temp.add(str(view)+'_'+str(index))
                            index+=1

                    #判断是否为最优的boundingbox集合
                    if object_img_temp:
                        flag=1
                        jj=0
                        while jj<len(boxes_index):
                            #提取已经保存的对象的boundingbox集合，跟当前三维点对应的boundingbox集合进行交集
                            if boxes_index_temp & boxes_index[jj]:#如果有交集，说明是同一个物体，要进行最优判断
                                if len(object_img_temp)>len(object_img[jj]):#如果当前boundingbox集合的长度大于之前保存的，说明当前的更优，替换
                                    boxes_index.pop(jj)
                                    object_img.pop(jj)
                                    jj-=1
                                else:#如果提取的子图数量不够之前的多，说明不是最优，不保存
                                    flag=0
                            jj+=1

                        #将符合要求的添加到总集合里
                        if flag==1:
                            object_img.append(object_img_temp)
                            boxes_index.append(boxes_index_temp)
        time_end=time.time()

        print('遍历消耗：',time_end-time_start)
        count=1
        for object_img_temp in object_img:
            name3D='object'+str(count)
            count2=1
            for i in object_img_temp:
                cv2.imwrite(dir_3D_path+'/'+name3D+'_'+str(count2)+'.jpg',i)
                count2+=1
            count+=1
        #view_point(img_name,ThreeD_point,camera_matrix,rotM1,tvec1,boxes.shape[0])
        #boxes_index是得到的boundingbox集合，可以调用进行任务调度

        '''
    elif mode == "dir_predict_super_resolution":#对图像进行自适应裁切后超分辨率，再重检测

        #超分辨率模型，图片识别可以用
        
        sr_model = EDSR().to('cuda' if torch.cuda.is_available() else 'cpu')
        sr_model.load_state_dict(torch.load('./model_data/visdrone_x2.pt'))
        

        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image,boxes,conf,classnames= yolo.detect_image(image)
                #r_image.show()
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

                #将拥挤区域提取
                image_cv2=cv2.imread(image_path, cv2.IMREAD_COLOR)
                corp_offsets, corp_imgs,crop_boxes=self_adaptive_crop(boxes, image_cv2)
                #超分
                sr_images=[]
                sr_num=1
                for corp_img in corp_imgs:
                    cv2.imwrite(dir_save_path+'/crop/'+str(sr_num)+'_crop_'+img_name,corp_img)#保存没有超分的子图像
                    sr_name='sr_'+str(sr_num)+'_'+img_name
                    cv2.imwrite(dir_redetect_path+'/'+sr_name,corp_img)

                    #超分辨率
                    sr_input = torch.FloatTensor(corp_img.copy()).to('cuda' if torch.cuda.is_available() else 'cpu').permute(2, 0, 1).unsqueeze(0)
                    image_data = sr_model(sr_input)
                    image_data = image_data.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
                    sr_images.append(image_data)
                    sr_name='sr_'+str(sr_num)+'_'+img_name
                    cv2.imwrite(dir_redetect_path+'/'+sr_name,image_data)

                    sr_num+=1

                    #重检测
                    image_path  = os.path.join(dir_redetect_path, sr_name)
                    image       = Image.open(image_path)
                    r_image,boxes,conf,classnames= yolo.detect_image(image)
                    #r_image.show()
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    r_image.save(os.path.join(dir_save_path, sr_name))
        '''
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")




