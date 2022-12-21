


import sys
#sys.path.append('../')
import random
import time
import os
import shutil
import numpy as np
import torch
from torch import nn
import queue
import threading
import argparse
from sklearn import linear_model
from module.Socketnet.Socketnet import Socketnet
from module.Detection.Detector import Detector
from module.Alignment.Alignmenter import Alignmenter
from module.Identification.Identifier import Identifier


def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

class Executor(object):
    def __init__(self,drone_id,CPU,GPU,device_number):
        self.drone_id=drone_id#无人机的ID
        self.CPU=CPU#当前设备的CPU性能
        self.GPU=GPU#当前设备的GPU性能
        self.device_number=device_number#所有设备的数目
        self.T=3#分配周期
        #加载模块
        self.socketnet=Socketnet(device_id=str(drone_id))#通信网络
        self.detector=Detector(self.drone_id)#检测器
        self.alignmenter=Alignmenter()#对齐器
        self.alignmenter.prepare()#初始化对齐器
        self.identifier=Identifier()#识别器
        #id与文件名的列表：
        self.id_name={0:'Cloud',1:'Drone1',2:'Drone2',3:'Drone3'}
        #加载IO路径
        #self.root_path='F:/MCOT/system/'+self.id_name[self.drone_id]
        self.root_path={0:'F:/MCOT/system/'+self.id_name[0],1:'/home/tyz/'+self.id_name[1],2:'/home/yhj1/'+self.id_name[2]}
        self.dataset_path='/dataset/'
        self.detect_result_path='/detect_result/'
        setDir(self.root_path[self.drone_id]+self.detect_result_path)
        self.to_allocate_path='/task_to_allocate/'
        setDir(self.root_path[self.drone_id]+self.to_allocate_path)
        self.tasks_path='/tasks/'
        for i in range(1,self.device_number):
            setDir(self.root_path[self.drone_id]+self.tasks_path+'bb_'+str(i)+'/')
        self.result_path='/final_result/'
        setDir(self.root_path[self.drone_id]+self.result_path)
        self.exp_path='/exp_pool/'
        setDir(self.root_path[self.drone_id]+self.exp_path)
        self.start_path='/start/'
        setDir(self.root_path[self.drone_id]+self.start_path)
        #已分配、执行列表
        self.already_allocate=[]
        self.already_execute=[]
        self.already_send=[]


    #文件写入
    def write_flie(self,filename, text):
        file = open(filename, 'w')
        file.write(str(text))
        file.close()
    #文件读取
    def read_flie(self,filename):
        # 获取文件内容
        try:
            file = open(filename, "r")
            text = file.readline()
        except IOError:
            time.sleep(0.1)
            file = open(filename, "r")
            text = file.readline()
        finally:
            file.close()
        try:
            a = float(text)
        except:
            a = 1
        return a
    #文件接收
    def recv(self):
        self.socketnet.recv()
    #延迟获取
    def get_trandelay(self):
        #向所有设备发送延迟文件
        for i in range(self.device_number):
            if i==self.drone_id:
                continue

            now_time = time.time()
            self.write_flie("delay/Delay_"+str(self.drone_id)+"_"+str(i)+".txt", now_time)
            self.socketnet.send(str(i),"delay/Delay_"+str(self.drone_id)+"_"+str(i)+".txt","delay/")

        #读取边缘节点传输时延，发送给云端
        for i in range(1,self.device_number):
            if i==self.drone_id:
                continue
            start_time = self.read_flie("delay/Delay_"+str(i)+"_"+str(self.drone_id)+".txt")
            end_time = os.path.getmtime("delay/Delay_"+str(i)+"_"+str(self.drone_id)+".txt")
            Delay_now=end_time-start_time
            self.write_flie("delay/Delay_"+str(i)+"_"+str(self.drone_id)+".txt", Delay_now)
            self.socketnet.send(str(0),"delay/Delay_"+str(i)+"_"+str(self.drone_id)+".txt","delay/")
    #获取待分配任务
    def select_to_allocate(self):
        # 待分配的任务时间戳
        to_allocate = []
        # print(img_dir)
        img_paths = os.listdir(self.root_path[self.drone_id]+self.detect_result_path)

        for img_path in img_paths:
            # 截取时间戳，start_index为_的下标的下一位，end_index为最后一个.的下标
            timestamp = img_path
            # 如果此时间戳的任务没有被分配，则加入到待分配中
            if timestamp not in self.already_allocate and timestamp not in to_allocate:
                to_allocate.append(timestamp)
                self.already_allocate.append(timestamp)
        return to_allocate
    #写入待分配任务
    def write_to_allocate(self,to_allocate):
        to_allocate_str = ''
        with open(self.root_path[self.drone_id]+self.to_allocate_path+'toAllocate.txt', 'w') as f:
            # 遍历每个待分配的时间戳文件
            for timestamp in to_allocate:
                to_allocate_str += timestamp + '\n'
            f.write(to_allocate_str)
            f.close()
    #读取分配的任务列表
    def read_arrangement(self):
        arrangement_dict = {}
        while True:
            try:
                with open(self.root_path[self.drone_id]+self.to_allocate_path+'arrangement.txt', 'r') as f:
                    pairs = f.readlines()
                    for timestamp_executor in pairs:
                        timestamp = timestamp_executor.split(' ')[0]
                        executor = timestamp_executor.split(' ')[1]
                        if executor.endswith('\n'):
                            executor = executor[:-1]
                        if not timestamp in self.already_send:
                            arrangement_dict[timestamp] = executor
                            self.already_send.append(timestamp)
                    break
                    #print(1)
            except:
                #print(0)
                continue

        return arrangement_dict
    #将文件发送给对应的执行者
    def arrange(self):
        arrangement_dict = self.read_arrangement()

        for timestamp in arrangement_dict.keys():
            executor = int(arrangement_dict[timestamp])
            #print(executor)
            self.socketnet.send(str(executor),self.root_path[self.drone_id]+self.detect_result_path+'bb_'+str(self.drone_id)+'_'+timestamp,self.root_path[executor]+self.tasks_path+'bb_'+str(drone_id)+'/')
        return arrangement_dict
    #无人机端的任务分配流程
    def allocate(self):
        while True:
            # 获取还没有被分配的时间戳
            to_allocate = self.select_to_allocate()
            #print(to_allocate)
            # 将to_allocate写入到文件中
            if not to_allocate==[]:
                self.write_to_allocate(to_allocate)
                # 将to_allocate_path文件发送到云端
                if self.drone_id==1:
                    self.socketnet.send('0',self.root_path[self.drone_id]+self.to_allocate_path+'toAllocate.txt',self.root_path[0]+self.to_allocate_path)
                # 读取arrangement.txt文件, 将图片分发给目的端
                time.sleep(self.T)
    #无人机端目标检测
    def detect(self):
        images=os.listdir(self.root_path[self.drone_id]+self.dataset_path)
        for image in images:
            print('detect:',image)
            t_now=image.split('.')[0].split('_')[1]
            self.detector.predict(self.root_path[self.drone_id]+self.dataset_path+image,self.root_path[self.drone_id]+self.detect_result_path,t_now)

    def if_path_exist(self,timestemp):
        while True:
            bb_names=os.listdir(self.root_path[self.drone_id]+self.tasks_path)
            flag_exist=1
            for bb_name in bb_names:
                if os.path.exists(self.root_path[self.drone_id]+self.tasks_path+bb_name+'/'+bb_name+'_'+str(timestemp)+'/'):
                    continue
                else:
                    flag_exist=0
            if flag_exist==0:
                continue
            else:
                break

    #无人机端对齐+识别任务执行
    def execute_task(self):
        while True:
            arrangement_dict=self.arrange()
            if str(self.drone_id) in arrangement_dict.values():
                time.sleep(3)
                s0=0
                for timestemp in arrangement_dict.keys():
                    if not timestemp in self.already_execute and arrangement_dict[timestemp]==str(self.drone_id):
                        while True:
                            self.if_path_exist(timestemp)
                            if self.socketnet.recv_flag==1:
                                time.sleep(1)
                                continue
                            else:
                                s=time.time()
                                f=open(self.root_path[self.drone_id]+self.exp_path+'exp'+str(self.drone_id)+'.txt','a')
                                flag=1
                                images=self.alignmenter.align(dir_save_path=self.root_path[self.drone_id]+self.tasks_path,time_stemp=timestemp)
                                result_file=open(self.root_path[self.drone_id]+self.result_path+'result.txt','a')
                                for i,object in enumerate(images):
                                    name,distance=self.identifier.multi_detect_frame(object,i+1,'arcfusion')
                                    print(timestemp,name,distance,file=result_file)
                                result_file.close()
                                self.already_execute.append(timestemp)
                                e=time.time()
                                s0=s0+e-s
                                print('execute:',timestemp,'time_cost:',s0)
                                print(timestemp,s0,file=f)
                                f.close()
                                if flag==1 and not self.drone_id==0:
                                    self.socketnet.send('0',self.root_path[self.drone_id]+self.exp_path+'exp'+str(self.drone_id)+'.txt',self.root_path[0]+self.exp_path)
                                break







if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-id","--drone_id",help="Drone id",default=1)
    ap.add_argument("-CPU","--CPU",help="CPU's Force",default=1)
    ap.add_argument("-GPU","--GPU",help="GPU's Force",default=4)
    ap.add_argument("-n","--device_number",help="Device number",default=3)
    args=vars(ap.parse_args())

    drone_id=args["drone_id"]
    CPU=args["CPU"]
    GPU=args["GPU"]
    device_number=args["device_number"]

    executer=Executor(drone_id=drone_id,CPU=CPU,GPU=GPU,device_number=device_number)
    executer.socketnet.recv()
    t_detect=threading.Thread(target=executer.detect)
    t_execute_task=threading.Thread(target=executer.execute_task)
    t_allocate=threading.Thread(target=executer.allocate)

    while True:
        if os.path.exists(executer.root_path[executer.drone_id]+executer.start_path+'start.txt'):
            print('start!!!!!!!!!!!!')
            #t_detect.setDaemon(True)
            t_detect.start()
            time.sleep(2)
            #t_allocate.setDaemon(True)
            t_allocate.start()
            #t_execute_task.setDaemon(True)
            t_execute_task.start()

            t_detect.join()
            t_allocate.join()
            t_execute_task.join()
            break
        else:
            continue






