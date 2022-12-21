import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['NUMBAPRO_CUDALIB']='C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\bin'
#from pyculib import blas
from PIL import ImageDraw
import math
import time
import cv2
import numpy as np
import argparse
from tqdm import tqdm


class Alignmenter(object):

    def __init__(self):
        self.path_3D='module/Alignment/3Dpoint.txt'
        self.path_camera='module/Alignment/drone.txt'

    def prepare(self):
        #读取文件夹中的3d点集合
        point3D_list=self.load_point(self.path_3D)
        #读取相机内参矩阵
        camera_point=self.load_point(self.path_camera)

        #解析3d点
        point3D_list=self.analysis_point3D(point3D_list)

        #解析相机内参
        camera_matrix,dist_coefs=self.analysis_camera(camera_point)

        self.camera_matrix=camera_matrix

        self.dist_coefs=dist_coefs

        #特征点世界坐标
        self.object_3d_points = np.array((point3D_list), dtype=np.double)

    def load_point(self,root_path):
        with open(root_path, "r") as f:
            point = f.readlines()
            for i in range(len(point)):
                point[i]=point[i].strip('\n')
        return point

    def analysis_point3D(self,point3D_list):
        for i in range(len(point3D_list)):
            p3D=point3D_list[i].split(',')
            for j in range(len(p3D)):
                p3D[j]=float(p3D[j])
            point3D_list[i]=p3D
        return point3D_list

    def analysis_camera(self,camera):
        camera_matrix=[]
        dist_coefs=[]
        for i in range(0,len(camera),5):
            camera_matrix_temp=[]
            dist_coefs_temp=[]
            for j in range(4):
                camera_point=camera[i+1+j].split(' ')
                if j==3:
                    dist_coefs_temp=camera_point
                else:
                    camera_matrix_temp.append(camera_point)
            dist_coefs_temp=np.array(dist_coefs_temp, dtype=np.double)
            try:
                camera_matrix_temp=np.array(camera_matrix_temp, dtype=np.double)
            except:
                print(1)
            camera_matrix.append(camera_matrix_temp)
            dist_coefs.append(dist_coefs_temp)
        return camera_matrix,dist_coefs

    def load_UAV_dir(self,dir_save_path):
        UAV_num=0
        UAV_paths_all=[]
        UAV_paths=os.listdir(dir_save_path)
        for UAV_path in UAV_paths:
            if UAV_path[0:2]=='bb':
                UAV_path2=os.listdir(dir_save_path+UAV_path+'/')
                UAV_path3=[]
                for t in range(len(UAV_path2)):
                    UAV_path3.append(UAV_path2[t][5::])
                    UAV_path2[t]=dir_save_path+UAV_path+'/'+UAV_path2[t]+'/'
                UAV_path_dict=dict(zip(UAV_path3,UAV_path2))
                UAV_paths_all.append(UAV_path_dict)
                UAV_num+=1
        return UAV_paths_all,UAV_num

    def load_UAV_data(self,bb_paths):
        imgnum=0
        boxes_allview=[]
        image_allview=[]
        size=[]
        object_2d_point=[]

        for bb_path in bb_paths:
            image_thisview=[]
            file_names = os.listdir(bb_path)
            for file_name in file_names:
                nowfile_path=bb_path+file_name
                if file_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):

                    continue
                elif file_name.split('.')[0]=='box':
                    bb_nowview=[]
                    txt_file=open(nowfile_path,'r')
                    for i,line in enumerate(txt_file):
                        if i==0:
                            size.append(line.split(' '))
                        else:
                            box_name=line.split(':')[0].split(' ')[0]
                            line=line.split(':')[1]
                            line=line.split(' ')
                            bb_nowview.append([int(line[1]),int(line[2]),int(line[3]),int(line[4].split('\\')[0])])
                            try:
                                image_thisview.append(cv2.imread(bb_path+box_name+'.jpg'))
                            except:
                                image_thisview.append(None)
                    boxes_allview.append(bb_nowview)
                else:
                    txt_file=open(nowfile_path,'r')
                    object_2d_point_temp=[]
                    for i,line in enumerate(txt_file):
                        line=line.split(' ')
                        for j in range(len(line)):
                            line[j]=int(line[j])
                        object_2d_point_temp.append(line)
                    object_2d_point.append(object_2d_point_temp)

            image_allview.append(image_thisview)
            imgnum+=1

        return boxes_allview,image_allview,imgnum,size,object_2d_point

    def RotateByZ(self,Cx, Cy, thetaZ):
        rz = thetaZ*math.pi/180.0
        outX = math.cos(rz)*Cx - math.sin(rz)*Cy
        outY = math.sin(rz)*Cx + math.cos(rz)*Cy
        return outX, outY
    def RotateByY(self,Cx, Cz, thetaY):
        ry = thetaY*math.pi/180.0
        outZ = math.cos(ry)*Cz - math.sin(ry)*Cx
        outX = math.sin(ry)*Cz + math.cos(ry)*Cx
        return outX, outZ
    def RotateByX(self,Cy, Cz, thetaX):
        rx = thetaX*math.pi/180.0
        outY = math.cos(rx)*Cy - math.sin(rx)*Cz
        outZ = math.sin(rx)*Cy + math.cos(rx)*Cz
        return outY, outZ

    def get_camera_position(self,object_3d_points,object_2d_point1,camera_matrix,dist_coefs):
        #求解相机位姿
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
        (x1, y1) = self.RotateByZ(x1, y1, -1*thetaZ1)
        (x1, z1) = self.RotateByY(x1, z1, -1*thetaY1)
        (y1, z1) = self.RotateByX(y1, z1, -1*thetaX1)
        #获得相机坐标系在世界坐标系的位置坐标
        Cx1 = x1*-1
        Cy1 = y1*-1
        Cz1 = z1*-1

        p_camera=[Cx1,Cy1,Cz1]

        return p_camera,thetaX1,thetaY1,thetaZ1,rotM1,tvec1

    def get_all_camera_position(self,object_3d_points,object_2d_point,camera_matrix,dist_coefs,imgnum):
        p_camera=[]
        thetaX=[]
        thetaY=[]
        thetaZ=[]
        rotM=[]
        tvec=[]
        #计算每一副视图的旋转变换矩阵
        for i in range(imgnum):
            p_camera1,thetaX1,thetaY1,thetaZ1,rotM1,tvec1=self.get_camera_position(object_3d_points,object_2d_point[i],camera_matrix[i],dist_coefs[i])
            p_camera.append(p_camera1)
            thetaX.append(thetaX1)
            thetaY.append(thetaY1)
            thetaZ.append(thetaZ1)
            rotM.append(rotM1)
            tvec.append(tvec1)

        return p_camera,thetaX,thetaY,thetaZ,rotM,tvec

    def inbox(self,p,box):
        if p[1]>=box[0] and p[1]<=box[2] and p[0]>=box[1] and p[0]<=box[3]:
            return 1
        else:
            return 0

    def center_distance(self,p,box):
        d=abs(p[1]-(box[0]+box[2])/2)**2+abs(p[0]-(box[1]+box[3])/2)**2
        return d
    def C2W(self,p,rotM,tvec):
        p=np.array(p).reshape(3,-1)
        matrix=np.matrix(rotM)
        # reverse T
        tmp=p-tvec
        # reverse R
        tmp2=np.dot(matrix.I,tmp)
        return np.append(tmp2,[[1]],axis=0)

    def T2D(self,p,camera_matrix,rotM,tvec):
        s=time.time()
        Out_matrix = np.concatenate((rotM, tvec), axis=1)
        pixel = np.dot(camera_matrix, Out_matrix)
        pixel1 = np.dot(pixel,p)
        #pixel = blas.gemm('N', 'N', 1, camera_matrix, Out_matrix)
        #pixel1 = blas.gemm('N', 'N', 1, pixel, p)
        pixel2 = pixel1/pixel1[2]
        pixel2 = pixel2.astype(np.int)
        e=time.time()
        #print(e-s)
        return pixel2

    def D2C(self,p,camera_matrix):
        camera_p=np.dot(np.linalg.inv(camera_matrix),p)
        return camera_p


    def alignment_p2(self,camera_matrix,rotM,tvec,imgnum,boxes_allview,image_allview):
        #开始对齐
        time_start=time.time()
        list_object_img=[]#所有对象的图片集合
        for view in range(imgnum):
            if view==0:
                list_points_in_boxes=[]
                flag=1
                zc=1
                step=100
                while True:
                    break_flag=0
                    for box_num,box in enumerate(boxes_allview[view]):
                        if flag==1:
                            list_points_in_boxes.append([])
                            list_object_img.append([image_allview[view][box_num]])
                        box_center=np.array([(box[1]+box[3])/2,(box[0]+box[2])/2,1])
                        center=(int(box_center[0]),int(box_center[1]))
                        #cv2.circle(test_img, center, 5, (0,0,255),0)
                        vector_center=np.expand_dims(np.array([box_center[0],box_center[1],1])*zc,axis=0).T
                        point_linepoint1=self.C2W(self.D2C(vector_center,camera_matrix[view]),rotM[view],tvec[view])
                        if point_linepoint1[2]<=250 and point_linepoint1[2]>=0:
                            step=10
                            list_points_in_boxes[box_num].append(point_linepoint1)
                        if point_linepoint1[2]<0:
                            break_flag+=1
                    flag=0
                    if break_flag==len(boxes_allview[view]):
                        break
                    zc+=step
                    #print(zc)
                for i in range(len(list_points_in_boxes)):
                    list_points_in_boxes[i].reverse()
                #cv2.imwrite('result1.jpg',test_img)


            else:
                #test_img=cv2.imread('2_1.jpg')
                for i,point_set in enumerate(list_points_in_boxes):
                    flag=1
                    for p in point_set:
                        piexl_now=self.T2D(np.array(p,dtype=np.double),camera_matrix[view],rotM[view],tvec[view])
                        center=(int(piexl_now[0]),int(piexl_now[1]))
                        #cv2.circle(test_img, center, 5, (0,0,255),0)
                        inbox_flag=0
                        box_distance=[]
                        for index,box in enumerate(boxes_allview[view]):
                            box_distance.append(self.center_distance(piexl_now,box))

                            if self.inbox(piexl_now,box):
                                inbox_flag=1
                        if inbox_flag==1:
                            #在框中则将其添加到当前三维点对应的图片集合与boundingbox集合中

                            min_dis=min(box_distance)
                            dis_index=box_distance.index(min_dis)
                            if min_dis<4000:
                                list_object_img[i].append(image_allview[view][dis_index])

                                flag=0
                        if flag==0:
                            break

        time_end=time.time()
        #print('搜索消耗：',time_end-time_start)
        return list_object_img,list_points_in_boxes


    def align(self,dir_save_path,time_stemp):

        #获取每个UAV传来的文件夹位置，UAV_paths_all包含每个无人机的字典，字典的key是时间戳，值是时间戳对应的数据，UAV_num为无人机的数目
        UAV_paths_all,UAV_num=self.load_UAV_dir(dir_save_path)

        key = str(time_stemp)
        #获取当前时间戳中每个无人机的数据地址，bb_paths即为每个无人机当前时间戳的数据地址的列表
        bb_paths=[]
        for i in range(UAV_num):
            bb_paths.append(UAV_paths_all[i][key])
        #print("time_stemp:",key)

        #根据数据地址加载数据，boxes_allview为各无人机当前帧的检测框列表，image_allview为各无人机当前帧的图像list，imgnum为图片数量（理论上会等于无人机的数量）
        boxes_allview,image_allview,imgnum,size,point2D_list=self.load_UAV_data(bb_paths)

        object_2d_point=[]
        for ptemp in point2D_list:
            object_2d_point_temp = np.array((ptemp), dtype=np.double)
            object_2d_point.append(object_2d_point_temp)

        #计算每一副视图的旋转变换矩阵、得到相机的位姿，各轴旋转角，以及内参外参矩阵
        p_camera,thetaX,thetaY,thetaZ,rotM,tvec=self.get_all_camera_position(self.object_3d_points,object_2d_point,self.camera_matrix,self.dist_coefs,imgnum)

        #对齐算法
        object_img,boxes_index=self.alignment_p2(self.camera_matrix,rotM,tvec,imgnum,boxes_allview,image_allview)

        #保存对齐结果
        return object_img

if __name__ == "__main__":
    alignmenter=Alignmenter()
    alignmenter.prepare()
    images=alignmenter.align(dir_save_path="align_bb_0/",time_stemp=1)
    print(1)


