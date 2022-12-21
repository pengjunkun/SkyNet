import sys
sys.path.append('../')
import random
import time
import os

import numpy as np
import torch
from torch import nn
import queue
import threading
from sklearn import linear_model
from .Socketnet.Socketnet import Socketnet

class Executor(object):
    def __init__(self,drone_id):
        self.drone_id=drone_id
        self.socketnet=Socketnet(device_id=drone_id)
        self.CPU=1
        self.GPU=8
        self.T=1000
        self.already_allocate=[]
        self.device_number=2

    def write_flie(self,filename, text):
        file = open(filename, 'w')
        file.write(str(text))
        file.close()

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

    def get_trandelay(self):
        #向所有设备发送延迟文件
        for i in range(self.device_number-1):
            if i==self.drone_id:
                continue

            now_time = time.time()
            self.write_flie("Delay_"+str(self.drone_id)+"_"+str(i)+".txt", now_time)
            self.socketnet.send(str(i),"Delay_"+str(self.drone_id)+"_"+str(i)+".txt","Delay_"+str(self.drone_id)+"_"+str(i)+".txt")

        #读取边缘节点传输时延，发送给云端
        for i in range(self.device_number-1):
            if i==self.drone_id:
                continue
            start_time = self.read_flie("Delay_"+str(i)+"_"+str(self.drone_id)+".txt")
            end_time = os.path.getmtime("Delay_"+str(i)+"_"+str(self.drone_id)+".txt")
            Delay_now=end_time-start_time
            self.write_flie("Delay_"+str(i)+"_"+str(self.drone_id)+".txt", Delay_now)
            self.socketnet.send(str(0),"Delay_"+str(i)+"_"+str(self.drone_id)+".txt","Delay_"+str(i)+"_"+str(self.drone_id)+".txt")

    def recv(self):
        self.socketnet.recv()


if __name__ == "__main__":
    executor=Executor(drone_id=0)
    executor.recv()
    executor.get_trandelay()



