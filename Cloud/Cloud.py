import sys
#sys.path.append('../')
import random
import time
import os
import numpy as np
import torch
from torch import nn
import shutil
import queue
import threading
import argparse
from sklearn import linear_model
from module.Socketnet.Socketnet import Socketnet
from module.Detection.Detector import Detector
from module.Alignment.Alignmenter import Alignmenter
from module.Identification.Identifier import Identifier

def setDir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

class Executor(object):
    def __init__(self,drone_id,CPU,GPU,device_number):
        self.drone_id=drone_id#The ID of the drone
        self.CPU=CPU#CPU performance of the current device
        self.GPU=GPU#GPU performance of the current device
        self.device_number=device_number#The number of all devices in system
        self.T=3#Task assignment cycle

        #System module
        self.socketnet=Socketnet(device_id=str(drone_id))#Communication network
        self.detector=Detector(self.drone_id)#detector
        self.alignmenter=Alignmenter()#Alignmenter
        self.alignmenter.prepare()#init Alignmenter
        self.identifier=Identifier()#识别器

        #Load the IO path
        self.id_name={0:'Cloud',1:'Drone1',2:'Drone2',3:'Drone3'}
        self.root_path={0:'H:/MCOT/system/'+self.id_name[0],1:'/home/tyz/'+self.id_name[1],2:'/home/yhj1/'+self.id_name[2]}
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
        #Assigned, executed list
        self.already_allocate=[]
        self.already_execute=[]
        self.already_send=[]


    #File write
    def write_flie(self,filename, text):
        file = open(filename, 'w')
        file.write(str(text))
        file.close()
    #File read
    def read_flie(self,filename):
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

    #File reception
    def recv(self):
        self.socketnet.recv()

    #Delayed acquisition
    def get_trandelay(self):
        #Send the delayed file to all devices
        for i in range(self.device_number):
            if i==self.drone_id:
                continue

            now_time = time.time()
            self.write_flie("delay/Delay_"+str(self.drone_id)+"_"+str(i)+".txt", now_time)
            self.socketnet.send(str(i),"delay/Delay_"+str(self.drone_id)+"_"+str(i)+".txt","delay/")

        #read transmission delay of the edge node and sent to the cloud
        for i in range(1,self.device_number):
            if i==self.drone_id:
                continue
            start_time = self.read_flie("delay/Delay_"+str(i)+"_"+str(self.drone_id)+".txt")
            end_time = os.path.getmtime("delay/Delay_"+str(i)+"_"+str(self.drone_id)+".txt")
            Delay_now=end_time-start_time
            self.write_flie("delay/Delay_"+str(i)+"_"+str(self.drone_id)+".txt", Delay_now)
            self.socketnet.send(str(0),"delay/Delay_"+str(i)+"_"+str(self.drone_id)+".txt","delay/")

    #Gets the task to be assigned
    def select_to_allocate(self):
        # Time stamp of the task to be assigned
        to_allocate = []
        # print(img_dir)
        img_paths = os.listdir(self.root_path[self.drone_id]+self.detect_result_path)

        for img_path in img_paths:
            timestamp = img_path
            # If the task with this timestamp is not assigned, it is added to the unassigned task
            if timestamp not in self.already_allocate and timestamp not in to_allocate:
                to_allocate.append(timestamp)
                self.already_allocate.append(timestamp)
        return to_allocate

    #Writes the task to be assigned
    def write_to_allocate(self,to_allocate):
        to_allocate_str = ''
        with open(self.root_path[self.drone_id]+self.to_allocate_path+'toAllocate.txt', 'w') as f:
            # Iterate through each timestamp file to be allocated
            for timestamp in to_allocate:
                to_allocate_str += timestamp + '\n'
            f.write(to_allocate_str)
            f.close()

    #Reads the assigned task list
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
            except:
                continue
        return arrangement_dict

    #Sends the file to the appropriate executor
    def arrange(self):
        arrangement_dict = self.read_arrangement()

        #for timestamp in arrangement_dict.keys():
            #executor = int(arrangement_dict[timestamp])
            #print(executor)
            #self.socketnet.send(str(executor),self.root_path[self.drone_id]+self.detect_result_path+'bb_'+str(self.drone_id)+'_'+timestamp,self.root_path[executor]+self.tasks_path+'bb_'+str(drone_id)+'/')
        return arrangement_dict

    #Drone side task assignment
    def allocate(self):
        while True:
            # Gets timestamps that have not yet been allocated
            to_allocate = self.select_to_allocate()
            # Write to_allocate to the file
            if not to_allocate==[]:
                self.write_to_allocate(to_allocate)
                # Send the to_allocate_path file to the cloud
                if self.drone_id==1:
                    self.socketnet.send('0',self.root_path[self.drone_id]+self.to_allocate_path+'toAllocate.txt',self.root_path[0]+self.to_allocate_path)
                # Read the arrangement.txt file and distribute the image to the destination end
                time.sleep(self.T)

    #Drone side detection
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

    #DD task
    def execute_task(self):
        while True:
            #Read the task assignment list
            arrangement_dict=self.arrange()
            #If this device is assigned a mission, start
            if str(self.drone_id) in arrangement_dict.values():
                s0=0
                for timestemp in arrangement_dict.keys():
                    #If the task is not executed
                    if not timestemp in self.already_execute and arrangement_dict[timestemp]==str(self.drone_id):
                        while True:
                            self.if_path_exist(timestemp)
                            #Wait for the data transfer to complete
                            if self.socketnet.recv_flag==1:
                                time.sleep(1)
                                continue
                            else:
                                s=time.time()
                                f=open(self.root_path[self.drone_id]+self.exp_path+'exp'+str(self.drone_id)+'.txt','a')
                                flag=1
                                #align
                                images=self.alignmenter.align(dir_save_path=self.root_path[self.drone_id]+self.tasks_path,time_stemp=timestemp)
                                result_file=open(self.root_path[self.drone_id]+self.result_path+'result.txt','a')
                                #identify
                                for i,object in enumerate(images):
                                    name,distance=self.identifier.multi_detect_frame(object,i+1,'arcfusion')
                                    print(timestemp,name,distance,file=result_file)
                                result_file.close()
                                #finish this task
                                self.already_execute.append(timestemp)
                                e=time.time()
                                s0=s0+e-s
                                print('execute:',timestemp,'time_cost:',s0)
                                print(timestemp,s0,file=f)
                                f.close()
                                if flag==1 and not self.drone_id==0:
                                    self.socketnet.send('0',self.root_path[self.drone_id]+self.exp_path+'exp'+str(self.drone_id)+'.txt',self.root_path[0]+self.exp_path)
                                break



class Cloud(Executor):
    def __init__(self,drone_id,CPU,GPU,device_number):
        Executor.__init__(self,drone_id=drone_id,CPU=CPU,GPU=GPU,device_number=device_number)
        #Randomly initialized linear model
        self.model = linear_model.LinearRegression()
        data = [[1, 5, 1, 4, 0.6],
                [1, 5, 2, 8, 0],
                [2, 10, 1, 4, 0.6],
                [2, 10, 2, 8, 0.1],
                [3, 15, 1, 4, 1.2]]
        data = np.array(data)
        x_train = data[:, :-1]
        x_train[:, 2:4] = 1.0 / x_train[:, 2:4]
        y_train = data[:, -1]
        self.model.fit(x_train, y_train)
        #hyperparameter
        self.epoch = 10
        self.batch_size = 1
        self.device_number=device_number
        self.drone_id=0
        #Load IO path
        self.id_name={0:'Cloud',1:'Drone1',2:'Drone2',3:'Drone3'}
        self.task_to_allocate_path='/task_to_allocate/'
        setDir(self.root_path[self.drone_id]+self.task_to_allocate_path)
        self.information_path='/information/'
        self.delay_path='/delay/'
        #Start signal
        f=open(self.root_path[self.drone_id]+self.start_path+'start.txt','w')
        f.close()


        #Task to be assigned
        self.timestamp_to_arrange = queue.Queue(maxsize=100)
        self.timestamp_to_TR_IM = {}
        #Assigned tasks and corresponding performers
        self.arrangement = {}
        #Transmission delay
        self.TM=np.zeros((device_number,device_number))
        #Experience playback pool
        self.pool_size=1000
        self.exp_poll=np.zeros((self.pool_size,5))
        self.count=0
        self.full_flag=0
        self.noresult_tasks={}


    #init information
    def init_information(self):
        with open(self.root_path[self.drone_id]+self.information_path+'arrangement.txt','w') as f:
            f.write('')
            f.close()
        with open(self.root_path[self.drone_id]+self.information_path+'TWLstart.txt', 'w') as f:
            f.write('2 0 0 0\n4 1 2 0\n3 1 2 0\n5 0 0 0\n1 3 6 0\n')
            f.close()
        with open(self.root_path[self.drone_id]+self.information_path+'TWLend.txt', 'w') as f:
            f.write('2 0\n4 1\n3 2\n5 0\n1 6\n')
            f.close()
        for i in range(self.device_number):
            filepath = self.root_path[self.drone_id]+self.information_path+'queue' + str(i) +'.txt'
            with open(filepath,'w') as f:
                f.write('0 0\n')
                f.close()

    #Read the task to be assigned
    def read_to_allocate_to_queue(self):
        if os.path.exists(self.root_path[self.drone_id]+self.task_to_allocate_path+"toAllocate.txt"):
            f=open(self.root_path[self.drone_id]+self.task_to_allocate_path+"toAllocate.txt",'r')
            timestamps = f.readlines()
            f.close()
            for timestamp in timestamps:
                if timestamp.endswith('\n'):
                    timestamp = timestamp[:-1]
                timestamp = int(timestamp.split('_')[2])
                if timestamp in self.arrangement.keys():
                    continue
                self.timestamp_to_arrange.put(timestamp)
        else:
            return

    def read_flie(self,filename):
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

    #Get the input to the model
    def single_sk_vector(self,target_id):
        queue_filename = self.root_path[self.drone_id]+self.information_path+'queue' + str(target_id) + '.txt'
        with open(queue_filename, 'r') as f:
            line = f.readlines()[-1]
            f.close()
        TR, IM = int(line.split(' ')[0]), int(line.split(' ')[1])
        xPU_file = open(self.root_path[self.drone_id]+self.information_path+str(target_id) + 'PUComputation.txt', 'r')
        xPU_lines = xPU_file.readlines()
        xPU_file.close()
        CPUi = int(xPU_lines[0].split(' ')[1])
        GPUi = int(xPU_lines[1].split(' ')[1])
        return np.array([TR, IM, CPUi, GPUi])

    #Predicted EFT
    def sk_vectors(self):
        vectors=[]
        for i in range(self.device_number):
            vectors.append(self.single_sk_vector(i))
        return vectors,self.model.predict(vectors)

    def write_TWLstart(self,now_task,TR,IM):
        with open(self.root_path[self.drone_id]+self.information_path+'TWLstart.txt', 'a') as f:
            w1 = str(now_task) + ' ' + str(TR) + ' ' + str(IM) + ' ' + str(time.time()) + '\n'
            f.write(w1)
            f.close()

    def write_TWLend(self,now_task,TR,IM):
        with open(self.root_path[self.drone_id]+self.information_path+'TWLend.txt', 'a') as f:
            w1 = str(now_task) + ' ' + str(time.time() + TR * 2 + IM * 3) + '\n'
            f.write(w1)
            f.close()

    def single_arrange(self,now_task,executor,exp_input):
        # Write task scheduling
        self.arrangement[now_task] = executor
        with open(self.root_path[self.drone_id]+self.information_path+'arrangement.txt', 'a') as f:
            f.write(str(now_task) + ' ' + str(executor) + '\n')
            f.close()
        # Read the queue_executor.txt file
        queue_txt_path = self.root_path[self.drone_id]+self.information_path+'queue' + str(executor) + '.txt'
        with open(queue_txt_path,'r') as f:
            line = f.readlines()[-1]
            TR = int(line.split(' ')[0])+1
            IM = int(line.split(' ')[1])+5
            f.close()
        with open(queue_txt_path,'a') as f:
            print(TR,IM,file=f)
            #if line == '':
                #return
            #TR = int(line.split(' ')[0])
            #IM = int(line.split(' ')[1])
            # 写timestamp TR IM start_time
            #self.write_TWLstart(now_task,TR,IM)
            #self.write_TWLend(now_task,TR,IM)
            f.close()
        self.noresult_tasks[now_task]=exp_input

    #Send task arrangement
    def send_arrangement(self):
        send_file_path=self.root_path[self.drone_id]+self.information_path+'arrangement.txt'
        for i in range(0,self.device_number):
            detination=self.task_to_allocate_path
            self.socketnet.send(str(i),send_file_path,self.root_path[i]+detination)

    #Model updating
    def model_update(self):
        if self.count==0:
            return
        print('start updateeeeeeeeeeeeeeee')
        if self.full_flag==1:
            data=self.exp_poll
        else:
            data=self.exp_poll[0:self.count]
        x_train = data[:, :-1]
        x_train[:, 2:4] = 1.0 / x_train[:, 2:4]
        y_train = data[:, -1]
        self.model.fit(x_train, y_train)

    #save exp_pool
    def exp_pool_input(self):
        exp_names=os.listdir(self.root_path[self.drone_id]+self.exp_path)
        if self.socketnet.recv_flag==2:
            time.sleep(1)
            exp_names=os.listdir(self.root_path[self.drone_id]+self.exp_path)
        if not len(exp_names) == 0:
            for exp_name in exp_names:
                f=open(self.root_path[self.drone_id]+self.exp_path+exp_name)
                lines=f.readlines()
                queue_filename = self.root_path[self.drone_id]+self.information_path+'queue' + exp_name.split('.')[0][-1] + '.txt'
                q_f=open(queue_filename,'r')
                now_queue=q_f.readlines()[-1].split(' ')
                now_task_num=int(now_queue[0])
                now_image_num=int(now_queue[1])
                q_f.close()

                for line in lines:
                    now_task=line.split(' ')
                    time_stemp=int(now_task[0])
                    wait_matrix=float(now_task[1])
                    if time_stemp in self.noresult_tasks:
                        wait_input=self.noresult_tasks[time_stemp]
                        x=np.append(wait_input, np.expand_dims(np.array(wait_matrix),axis=0) )
                        self.exp_poll[self.count]=x
                        if self.count+1==self.pool_size:
                            self.full_flag=1
                        self.count=(self.count+1)%self.pool_size
                        self.noresult_tasks.pop(time_stemp)
                        now_task_num=now_task_num-1
                        now_image_num=now_image_num-1
                    else:
                        continue
                f.close()
                q_f=open(queue_filename,'a')
                print(now_task_num,now_image_num,file=q_f)
                q_f.close()


    #Task allocation
    def task_arrangement(self):
        counter=0
        while True:
            self.read_to_allocate_to_queue()
            #Calculated transmission delay
            while self.timestamp_to_arrange.empty() == False:
                print(self.timestamp_to_arrange)
                for i in range(1,self.device_number):
                    for j in range(self.device_number):
                        if i==j:
                            continue
                        Delay=self.read_flie(self.root_path[self.drone_id]+self.delay_path+"Delay_"+str(i)+"_"+str(j)+".txt")
                        if j==0:
                            now_time=time.time()
                            self.TM[j][i]=self.TM[i][j]=now_time-Delay
                        else:
                            self.TM[i][j]
                #Get the predicted wait time and its input
                wait_input,wait_matrix = self.sk_vectors()
                #Gets the maximum transfer latency
                Delay_matirx = np.max(self.TM, axis=0)
                #Acquired total delay
                cost_matrix = Delay_matirx + wait_matrix
                executor=np.argmin(cost_matrix)
                exp_input=wait_input[executor]
                now_task = self.timestamp_to_arrange.get()
                self.single_arrange(now_task,executor,exp_input)
                self.send_arrangement()
                counter+=1
                if counter%self.T==0:
                    self.exp_pool_input()
                    self.model_update()


if __name__ == "__main__":

    ap=argparse.ArgumentParser()
    ap.add_argument("-id","--drone_id",help="Drone id",default=0)
    ap.add_argument("-CPU","--CPU",help="CPU's Force",default=3)
    ap.add_argument("-GPU","--GPU",help="GPU's Force",default=8)
    ap.add_argument("-n","--device_number",help="Device number",default=3)
    args=vars(ap.parse_args())

    drone_id=args["drone_id"]
    CPU=args["CPU"]
    GPU=args["GPU"]
    device_number=args["device_number"]


    cloud=Cloud(drone_id=drone_id,CPU=CPU,GPU=GPU,device_number=device_number)
    cloud.init_information()

    #Receiving thread
    cloud.socketnet.recv()

    #Task assignment thread
    t1=threading.Thread(target=cloud.task_arrangement)
    #DD end task thread
    t2=threading.Thread(target=cloud.execute_task)

    #Send to the other drones to start
    for i in range(1,device_number):
        cloud.socketnet.send(str(i),cloud.root_path[cloud.drone_id]+cloud.start_path+'start.txt',cloud.root_path[i]+cloud.start_path)

    t1.start()
    t2.start()


    t1.join()
    t2.join()






