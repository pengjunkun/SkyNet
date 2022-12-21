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


#-------------------------------------------------------计算三轴旋转角---------------------------------------------------#
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

#-------------------------------------------------------求解一个视图的相机位姿与旋转变换矩阵---------------------------------------------------#
def get_camera_position(object_3d_points,object_2d_point1,camera_matrix,dist_coefs):
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
    (x1, y1) = RotateByZ(x1, y1, -1*thetaZ1)
    (x1, z1) = RotateByY(x1, z1, -1*thetaY1)
    (y1, z1) = RotateByX(y1, z1, -1*thetaX1)
    #获得相机坐标系在世界坐标系的位置坐标
    Cx1 = x1*-1
    Cy1 = y1*-1
    Cz1 = z1*-1

    p_camera=[Cx1,Cy1,Cz1]

    return p_camera,thetaX1,thetaY1,thetaZ1,rotM1,tvec1

#-------------------------------------------------------将图像坐标系转化到相机坐标系---------------------------------------------------#
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

#-------------------------------------------------------三维点投影到二维图像点---------------------------------------------------#

def T2D(p,camera_matrix,rotM,tvec):
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

#-------------------------------------------------------三维点投影到二维图像点,返回图像坐标系---------------------------------------------------#

def T2D_row(p,camera_matrix,rotM,tvec):
    Out_matrix = np.concatenate((rotM, tvec), axis=1)
    pixel = np.dot(camera_matrix, Out_matrix)
    pixel1 = np.dot(pixel,p)
    return pixel1
#-------------------------------------------------------相机坐标系到图片坐标系---------------------------------------------------#
def C2D(p,camera_matrix):
    camera_p=np.dot(camera_matrix,p)
    return camera_p

#-------------------------------------------------------相机坐标系到世界坐标系---------------------------------------------------#
def C2W(p,rotM,tvec):
    p=np.array(p).reshape(3,-1)
    matrix=np.matrix(rotM)
    # reverse T
    tmp=p-tvec
    # reverse R
    tmp2=np.dot(matrix.I,tmp)
    return np.append(tmp2,[[1]],axis=0)

#-------------------------------------------------------相机坐标系到图片坐标系---------------------------------------------------#
def D2C(p,camera_matrix):
    camera_p=np.dot(np.linalg.inv(camera_matrix),p)
    return camera_p

#------------------------------------------------------返回投影矩阵---------------------------------------------------#
def matrix_T2D(camera_matrix,rotM,tvec):
    Out_matrix = np.concatenate((rotM, tvec), axis=1)
    pixel = np.dot(camera_matrix, Out_matrix)
    return pixel

#-------------------------------------------------------在图像上绘制投影点---------------------------------------------------#
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

#-------------------------------------------------------在图像上绘制预测框---------------------------------------------------#
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

#-------------------------------------------------------判断当前点是否在框内---------------------------------------------------#
def inbox(p,box):
    if p[1]>=box[0] and p[1]<=box[2] and p[0]>=box[1] and p[0]<=box[3]:
        return 1
    else:
        return 0

#-------------------------------------------------------计算当前点与坐标中心的距离---------------------------------------------------#
def center_distance(p,box):
    d=abs(p[1]-(box[0]+box[2])/2)**2+abs(p[0]-(box[1]+box[3])/2)**2
    return d


#-------------------------------------------------------加载txt文件中的特征点---------------------------------------------------#
def load_point(root):
    with open(root, "r") as f:
        point = f.readlines()
        for i in range(len(point)):
            point[i]=point[i].strip('\n')
    return point

#-------------------------------------------------------解析旋转变换矩阵---------------------------------------------------#
def analysis_camera(camera):
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

#-------------------------------------------------------解析三维特征点---------------------------------------------------#
def analysis_point3D(point3D_list,point_num):
    for i in range(point_num):
        p3D=point3D_list[i].split(',')
        for j in range(len(p3D)):
            p3D[j]=float(p3D[j])
        point3D_list[i]=p3D
    return point3D_list

#-------------------------------------------------------解析二维特征点---------------------------------------------------#
def analysis_point2D(point2D,point_num):
    point2D_list=[]
    for i in range(0,len(point2D),point_num+1):
        point2D_temp=[]
        for j in range(point_num):
            p2D=point2D[i+1+j].split(',')
            for k in range(len(p2D)):
                p2D[k]=float(p2D[k])
            point2D_temp.append(p2D)
        point2D_list.append(point2D_temp)
    return point2D_list

#-------------------------------------------------------加载无人机传来的文件地址---------------------------------------------------#
def load_UAV_dir(dir_save_path):
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

#-------------------------------------------------------根据地址加载由无人机传来的数据---------------------------------------------------#
def load_UAV_data(bb_paths):
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
                        a=bb_path+box_name+'.jpg'
                        image_thisview.append(cv2.imread(bb_path+box_name+'.jpg'))
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

#-------------------------------------------------------计算所有视图的位姿、旋转角、外参矩阵---------------------------------------------------#
def get_all_camera_position(object_3d_points,object_2d_point,camera_matrix,dist_coefs,imgnum):
    p_camera=[]
    thetaX=[]
    thetaY=[]
    thetaZ=[]
    rotM=[]
    tvec=[]
    #计算每一副视图的旋转变换矩阵
    for i in range(imgnum):
        p_camera1,thetaX1,thetaY1,thetaZ1,rotM1,tvec1=get_camera_position(object_3d_points,object_2d_point[i],camera_matrix[i],dist_coefs[i])
        p_camera.append(p_camera1)
        thetaX.append(thetaX1)
        thetaY.append(thetaY1)
        thetaZ.append(thetaZ1)
        rotM.append(rotM1)
        tvec.append(tvec1)

    return p_camera,thetaX,thetaY,thetaZ,rotM,tvec

#-------------------------------------------------------保存对齐的结果---------------------------------------------------#
def save_result_img(object_img,key):
    count=1
    for object_img_temp in object_img:
        name3D='object'+str(count)
        temp_save_dir=dir_3D_path+str(key)+'/'+name3D
        if not os.path.exists(temp_save_dir):
            os.makedirs(temp_save_dir)
        count2=1
        for i in object_img_temp:
            cv2.imwrite(temp_save_dir+'/'+str(count2)+'.jpg',i)
            count2+=1
        count+=1
#-------------------------------------------------------空间直线求解---------------------------------------------------#
def space_line(x,p1,p2):
    t=(x-p1[0])/(p2[0]-p1[0])
    y=t*(p2[1]-p1[1])+p1[1]
    z=t*(p2[2]-p1[2])+p1[2]
    p=np.array([x,y,z,1])
    return p

#-------------------------------------------------------对齐算法函数---------------------------------------------------#
def alignment_p0(camera_matrix,rotM,tvec,imgnum,boxes_allview,image_allview):
    #开始对齐
    time_start=time.time()
    object_img=[]#所有对象的图片集合
    boxes_index=[]#所有对象的boundingbox集合
    #三维点空间的遍历
    for i in tqdm(range(-1000,1000,10)):
        for j in range(-1000,1000,10):
            for k in range(100,250,10):
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
    return object_img,boxes_index
#-------------------------------------------------------对齐算法函数_process1---------------------------------------------------
def alignment_p1(camera_matrix,rotM,tvec,imgnum,boxes_allview,image_allview):
    #开始对齐
    time_start=time.time()
    list_object_img=[]#所有对象的图片集合
    #三维点空间的遍历
    for view in range(imgnum):
        if view==0:
            list_boxes_center_of_view1=[]
            list_points_of_boxes=[]
            for box in boxes_allview[view]:
                list_object_img.append([image_allview[view][box[0]:box[2],box[1]:box[3],:]])
                list_points_of_boxes.append([])
                list_boxes_center_of_view1.append(((box[1]+box[3])/2,(box[0]+box[2])/2))
            list_boxes_num=list(range(0,10))
            dict_points_of_boxes=dict(zip(list_boxes_num,list_points_of_boxes))
            for i in tqdm(range(-1000,1000,12)):
                for j in range(-1000,1000,12):
                    #for k in range(30,100,10):
                    p=[i,j,100,1]#当前三维点
                    #获得三维点在二维图像的投影位置
                    piexl_now=T2D(np.array(p,dtype=np.double),camera_matrix,rotM[view],tvec[view])
                    #判断是否在框中
                    for index,box in enumerate(boxes_allview[view]):
                        if inbox(piexl_now,box):
                            if abs(piexl_now[0]-list_boxes_center_of_view1[index][0])<30 and abs(piexl_now[1]-list_boxes_center_of_view1[index][1])<30:
                                #在框中则将其添加到当前三维点对应的图片集合与boundingbox集合中
                                dict_points_of_boxes[index].append(p)
        else:
            for i,point_set in enumerate(dict_points_of_boxes.values()):
                flag=1
                for p in point_set:
                    piexl_now=T2D(np.array(p,dtype=np.double),camera_matrix,rotM[view],tvec[view])
                    for index,box in enumerate(boxes_allview[view]):
                        if inbox(piexl_now,box):
                            #在框中则将其添加到当前三维点对应的图片集合与boundingbox集合中
                            list_object_img[i].append(image_allview[view][box[0]:box[2],box[1]:box[3],:])
                            #list_points_in_boxes[i]=[p]
                            flag=0
                    if flag==0:
                        break

    time_end=time.time()
    print('遍历消耗：',time_end-time_start)
    return list_object_img,dict_points_of_boxes

#-------------------------------------------------------对齐算法函数_process2---------------------------------------------------
def alignment_p2(camera_matrix,rotM,tvec,imgnum,boxes_allview,image_allview):
    #开始对齐
    time_start=time.time()
    list_object_img=[]#所有对象的图片集合
    for view in range(imgnum):
        if view==0:
            #test_img=cv2.imread('1_1.jpg')
            #能投影到第一个view的box的三维点集合
            list_points_in_boxes=[]
            flag=1
            zc=1
            step=100
            #for zc in tqdm(range(1,4000,10)):
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
                    point_linepoint1=C2W(D2C(vector_center,camera_matrix[view]),rotM[view],tvec[view])
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
                    piexl_now=T2D(np.array(p,dtype=np.double),camera_matrix[view],rotM[view],tvec[view])
                    center=(int(piexl_now[0]),int(piexl_now[1]))
                    #cv2.circle(test_img, center, 5, (0,0,255),0)
                    inbox_flag=0
                    box_distance=[]
                    for index,box in enumerate(boxes_allview[view]):
                        box_distance.append(center_distance(piexl_now,box))

                        if inbox(piexl_now,box):
                            inbox_flag=1
                    if inbox_flag==1:
                        #在框中则将其添加到当前三维点对应的图片集合与boundingbox集合中

                        min_dis=min(box_distance)
                        dis_index=box_distance.index(min_dis)
                        if min_dis<4000:
                            #box=boxes_allview[view][dis_index]
                            #p[2]=40
                            #test_flag=1
                            #for test_time in range(4):
                                #p[2]+=10
                                #piexl_now=T2D(np.array(p,dtype=np.double),camera_matrix,rotM[view],tvec[view])
                                #if inbox(piexl_now,box):
                                    #continue
                                #else:
                                    #test_flag=0
                            #if test_flag==1:
                            list_object_img[i].append(image_allview[view][dis_index])

                            flag=0
                    if flag==0:
                        break
            #cv2.imwrite('result2.jpg',test_img)

    time_end=time.time()
    print('搜索消耗：',time_end-time_start)
    return list_object_img,list_points_in_boxes

#-------------------------------------------------------判断投影点是否在图像内---------------------------------------------------#
def is_in_image(img,pixel):
    shape=img.shape
    if pixel[0]<0 or pixel[1]<0 or pixel[0]>shape[0] or pixel[1]>shape[1]:
        return False
    else:
        return True

#-------------------------------------------------------main---------------------------------------------------#
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-t","--time",help="time stemp",default=1)
    args=vars(ap.parse_args())
    #----------------------------------------------------------------------------------------------------------#
    UAV_id=1
    #   UAV_id用于指定无人机的编号
    #----------------------------------------------------------------------------------------------------------#
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_3D_path为对齐后按组分好的输出的图片路径
    #----------------------------------------------------------------------------------------------------------#
    dir_save_path   = "align_bb_0/"
    dir_3D_path = "align_bb_0/Align_result/"
    #创造路径
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)
    if not os.path.exists(dir_3D_path):
        os.makedirs(dir_3D_path)
    #----------------------------------------------------------------------------------------------------------#

    #读取文件中的2d点集合
    #point2D=load_point(dir_save_path+"2Dpoint.txt")
    #读取文件夹中的3d点集合
    point3D_list=load_point(dir_save_path+"3Dpoint.txt")
    #读取相机内参矩阵
    camera_point=load_point(dir_save_path+"drone.txt")

    point_num=len(point3D_list)
    #解析3d点
    point3D_list=analysis_point3D(point3D_list,point_num)

    #解析2d点
    #point2D_list=analysis_point2D(point2D,point_num)

    #解析相机内参
    camera_matrix,dist_coefs=analysis_camera(camera_point)

    #特征点世界坐标
    object_3d_points = np.array((point3D_list), dtype=np.double)

    #特征点在图的位置
    #object_2d_point=[]
    #for ptemp in point2D_list:
        #object_2d_point_temp = np.array((ptemp), dtype=np.double)
        #object_2d_point.append(object_2d_point_temp)

    #获取每个UAV传来的文件夹位置，UAV_paths_all包含每个无人机的字典，字典的key是时间戳，值是时间戳对应的数据，UAV_num为无人机的数目
    UAV_paths_all,UAV_num=load_UAV_dir(dir_save_path)


    key = str(args["time"])
    #获取当前时间戳中每个无人机的数据地址，bb_paths即为每个无人机当前时间戳的数据地址的列表
    bb_paths=[]
    for i in range(UAV_num):
        bb_paths.append(UAV_paths_all[i][key])
    print("time_stemp:",key)

    #根据数据地址加载数据，boxes_allview为各无人机当前帧的检测框列表，image_allview为各无人机当前帧的图像list，imgnum为图片数量（理论上会等于无人机的数量）
    boxes_allview,image_allview,imgnum,size,point2D_list=load_UAV_data(bb_paths)

    object_2d_point=[]
    for ptemp in point2D_list:
        object_2d_point_temp = np.array((ptemp), dtype=np.double)
        object_2d_point.append(object_2d_point_temp)

    #计算每一副视图的旋转变换矩阵、得到相机的位姿，各轴旋转角，以及内参外参矩阵
    p_camera,thetaX,thetaY,thetaZ,rotM,tvec=get_all_camera_position(object_3d_points,object_2d_point,camera_matrix,dist_coefs,imgnum)

    #对齐算法
    object_img,boxes_index=alignment_p2(camera_matrix,rotM,tvec,imgnum,boxes_allview,image_allview)

    #保存对齐结果
    save_result_img(object_img,key)





