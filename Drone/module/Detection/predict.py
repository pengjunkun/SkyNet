import torch
from PIL import ImageDraw, ImageFont
import math
import time
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
import yolo_barricade
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor,wait,ALL_COMPLETED
#from insightface.app import FaceAnalysis

def detect(yolo_now,image):
    r_image,boxes,conf,classnames = yolo_now.detect_image(image)
    return r_image,boxes,conf,classnames


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-m","--mode",help="Work mode",default="predict")
    ap.add_argument("-U","--UAV_ID",help="UAV's ID",default=3)
    ap.add_argument("-c","--camera",help="Camera ID",default=0)
    ap.add_argument("-I","--IMG",help="Image path",default='3_1.jpg')
    args=vars(ap.parse_args())

    yolo = YOLO()
    yolo_b = yolo_barricade.YOLO()
    yolo_b.detect_image(Image.open('warm_up.jpg'))
    yolo.detect_image(Image.open('warm_up.jpg'))

    #线程池
    pool = ThreadPoolExecutor(max_workers=5)
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    #mode = "dir_predict"
    mode =args["mode"]
    UAV_id=args["UAV_ID"]
    print("Work mode:",mode)
    print("UAV's ID:",UAV_id)
    #----------------------------------------------------------------------------------------------------------#
    #UAV_id=1
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
    video_path      = int(args["camera"])
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
    #app = FaceAnalysis()
    #app.prepare(ctx_id=0, det_size=(640, 640))



    if mode == "predict":
        #1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。
        #2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        #3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        #在原图上利用矩阵的方式进行截取。
        #4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        #比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。

        img =args["IMG"]
        s=time.time()
        image = Image.open(dir_origin_path+img)
        image_data=np.array(image)[...,::-1]


        thr1=pool.submit(detect,yolo,image)
        thr2=pool.submit(detect,yolo_b,image)

        r_image,boxes,conf,classnames=thr1.result()
        r_image2,boxes2,conf2,classnames2=thr2.result()

        now_time=img[2:-4]
        sub_save_path=dir_save_path+'bb_'+str(UAV_id)+'/'+'bb_'+str(UAV_id)+'_'+str(now_time)+'/'
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)
        txt_file_path = sub_save_path +  'box.txt'
        txt_file = open(txt_file_path,'w')
        print(image.size[0],image.size[1],file=txt_file)
        index=1
        e=time.time()
        print("检测消耗：",e-s)

        #s=time.time()
        for i,box in enumerate(boxes) :
            object_img=image_data[box[0]:box[2],box[1]:box[3],:]
            '''
            face_shape=object_img.shape
            face_bboxes, kpss = app.det_model.detect(object_img,max_num=0,metric='default')
            #faces = app.get(object_img)
            if not (face_bboxes.shape[0] ==0):
                face_bbox = face_bboxes[0, 0:4]
                w=max(face_bbox[2]-face_bbox[0],face_bbox[3]-face_bbox[1])
                center=[(face_bbox[2]+face_bbox[0])/2,(face_bbox[3]+face_bbox[1])/2]

                face_bbox[0]=max(0,center[0]-w/2)
                face_bbox[1]=max(0,center[1]-w/2)
                face_bbox[2]=min(face_shape[1],center[0]+w/2)
                face_bbox[3]=min(face_shape[0],center[1]+w/2)
                face_bbox=face_bbox.astype('int')
                face_img=object_img[face_bbox[1]:face_bbox[3],face_bbox[0]:face_bbox[2],:]
                #face_img=image_data[box[0]+face_bbox[1]:box[0]+face_bbox[3],box[1]+face_bbox[0]:box[1]+face_bbox[2],:]
                '''

            cv2.imwrite(sub_save_path+str(UAV_id)+'_'+str(index)+'.jpg',object_img)
            box_id= str(UAV_id)+'_'+str(index)
            #print(box_id,':',box[0]+face_bbox[1],box[1]+face_bbox[0],box[0]+face_bbox[3],box[1]+face_bbox[2],file=txt_file)
            print(box_id,':',box[0],box[1],box[2],box[3],file=txt_file)
            index+=1
        txt_file.close()

        txt_file_path = sub_save_path +  'featurepoint.txt'
        txt_file = open(txt_file_path,'w')
        dic_point=dict()
        for i,box in enumerate(boxes2) :
            center=(int((box[1]+box[3])/2),int((box[0]+box[2])/2))
            label= classnames2[i][-1:]
            dic_point[label]=center
            #object_img=image_data[box[0]:box[2],box[1]:box[3],:]
            #cv2.imwrite(sub_save_path+str(UAV_id)+'_'+str(index)+'.jpg',object_img)
            #index+=1
        for i in range(len(boxes2)):
            print(dic_point[str(i)][0],dic_point[str(i)][1],file=txt_file)
        txt_file.close()

        #e=time.time()
        #print("人脸检测消耗：",e-s)


    elif mode == "predict_single":
        s=time.time()
        img =args["IMG"]

        image = Image.open(dir_origin_path+img)
        image_data=np.array(image)[...,::-1]
        r_image,boxes,conf,classnames = yolo.detect_image(image)
        r_image.save(dir_save_path+img)
        now_time=img[2:-4]
        sub_save_path=dir_save_path+str(now_time)+'/'
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)
        for i,box in enumerate(boxes) :
            object_img=image_data[box[0]:box[2],box[1]:box[3],:]
            sub_save_path_obj=sub_save_path+'object'+str(i+1)+'/'
            if not os.path.exists(sub_save_path_obj):
                os.makedirs(sub_save_path_obj)
            cv2.imwrite(sub_save_path_obj+'1.jpg',object_img)
        e=time.time()
        print("时间消耗：",e-s)




    elif mode == "video":#视频检测模式
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
            row_frame=frame
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            # 进行检测
            frame,boxes,conf,classnames=yolo.detect_image(frame)

            #保存图片与box
            now_time=time.time()
            sub_save_path=dir_save_path+'bb_'+str(UAV_id)+'/'+'bb_'+str(UAV_id)+'_'+str(now_time)+'/'
            img_name=str(img_index)+'.jpg'
            img_index+=1

            os.makedirs(sub_save_path)
            cv2.imwrite(os.path.join(sub_save_path, img_name),row_frame)

            txt_file_path = sub_save_path +'box.txt'
            txt_file = open(txt_file_path,'w')
            index=1
            for box in boxes:
                box_id= str(UAV_id)+'_'+str(index)
                print(box_id,':',box[0],box[1],box[2],box[3],file=txt_file)
                index+=1
            txt_file.close()

            #在窗口实时显示视频结果
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
        img = Image.open('img/view1.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":#批量图片检测,模拟多个无人机

        img_names = os.listdir(dir_origin_path)
        now_time=time.time()
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                image_data=np.array(image)[...,::-1]
                r_image,boxes,conf,classnames= yolo.detect_image(image)
                r_image.show()
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

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")




