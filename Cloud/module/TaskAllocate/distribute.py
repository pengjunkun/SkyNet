import random
import time
import os

import numpy as np
import torch
from torch import nn
import queue
import threading
from sklearn import linear_model

yhj = False

user_names = {}

if yhj:
    user_names['0'] = 'cloud'
    user_names['1'] = 'yhj1'
    user_names['2'] = 'yhj2'
else:
    user_names['0'] = 'tzy/sjj'
    user_names['1'] = 'tzy/sjj'
    user_names['2'] = 'tzy/sjj'

TD_path_cloud = '/home/'+user_names['0']+'/TD_cloud+3/'
TD_path_1 = '/home/'+user_names['1']+'/TD_cloud+3/'
TD_path_2 = '/home/'+user_names['2']+'/TD_cloud+3/'

TD_paths = {
    '0':TD_path_cloud,
    '1':TD_path_1,
    '2':TD_path_2
}

# send.py文件位置
send_py_path = '/home/'+user_names['0']+'/TD_cloud+3/cloud/net/send.py'
# 无人机数量
drone_num = 2
now_execution = 0  # 0:cloud 1:一号无人机 2：二号无人机 3：三号无人机
deadline = 0  # 截止时间
Delay_1_2 = 0  # 传输延迟_1号无人机_2号无人机
Delay_1_3 = 0  # 传输延迟_1号无人机_3号无人机
Delay_2_3 = 0  # 传输延迟_2号无人机_3号无人机
Delay_1_cloud = 0  # 传输延迟_1号无人机_云端
Delay_2_cloud = 0  # 传输延迟_2号无人机_云端
Delay_3_cloud = 0  # 传输延迟_3号无人机_云端
wait_1 = 0  # 1号无人机等待时间
wait_2 = 0  # 2号无人机等待时间
wait_3 = 0  # 3号无人机等待时间
wait_cloud = 0  # 云端等待时间

def init():
    with open('arrangement.txt','w') as f:
        f.write('')
    with open('TWLstart.txt', 'w') as f:
        f.write('2 0 0 0\n4 1 2 0\n3 1 2 0\n5 0 0 0\n1 3 6 0\n')
    with open('TWLend.txt', 'w') as f:
        f.write('2 0\n4 1\n3 2\n5 0\n1 6\n')
    for i in range(drone_num):
        filepath = 'queue' + str(i+1) +'.txt'
        with open(filepath,'w') as f:
            f.write('0 0\n')

# 回归模型---求解等待时间ELTi
class LinearRegression(nn.Module):
    '''
        第i执行者队执行列预期等待时间：ELTi=f（TR，IN，CPU算力，GPU算力）
        f（TR，IN，CPU算力，GPU算力）使用机器学习回归模型求解
            TRi第i执行者表示目前等待执行的任务数
            INi第i执行者表示目前等待执行的图片数
            CPUi第i执行者的CPU算力---单位: TOPS
            GPUi第i执行者的GPU算力---单位: TOPS
        '''

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        # print(x.shape)
        # print(x)
        out = self.linear(x)
        return out


# 超参数a,b

def read_flie(filename):
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


def write_flie(filename, text):
    file = open(filename, 'w')
    file.write(str(text))
    file.close()

def send_arrangement():
    # 向各个无人机同步任务安排
    # send_file_path = '/home/linpeng/new/unknown_pictures/TD_cloud+3/drone_3/arrangement.txt'
    # destination = "/home/linpeng/new/unknown_pictures/TD_cloud+3/cloud"
    # arrangement.txt 分配方案文件的地址
    send_file_path = os.path.abspath('arrangement.txt')
    for i in range(drone_num):
        # destination = 'D:\\DevProjects\\python\\MCOT_nano\\sjj\\TD_cloud + 3\\drone_'+ str(i+1) +'\\任务分配\\'
        destination = TD_paths[str(i+1)] + 'drone_'+str(i+1)+'/taskAllocate/'
        order = "python3 "+ send_py_path + " " + str(i + 1) + " " + send_file_path + " " + destination
        # print(order)
        os.system(order)


# 数据迭代器
# def data_iter(batch_size,features,labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)
#     for i in range(0,num_examples,batch_size):
#         j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
#         yield features.index_select(0,j), labels.index_select(0,j)

def output():
    print("Delay_1_2:" + str(Delay_1_2))
    print("Delay_1_3:" + str(Delay_1_3))
    print("Delay_2_3:" + str(Delay_2_3))
    print("Delay_1_cloud:" + str(Delay_1_cloud))
    print("Delay_2_cloud:" + str(Delay_2_cloud))
    print("Delay_3_cloud:" + str(Delay_3_cloud))
    print("wait_1:" + str(wait_1))
    print("wait_2:" + str(wait_2))
    print("wait_3:" + str(wait_3))
    print("wait_cloud:" + str(wait_cloud))

def update_parameters(model, criterion, optimizer, device):
    # 时间周期T
    T = 10
    while True:
        # 读取数据
        try:
            data = union_TWL()
            if len(data) == 0:
                return
            data = np.array(data)
            features = torch.from_numpy(data[:, :-1]).float()
            features[:, 2:4] = 1.0 / features[:, 2:4]
            labels = torch.from_numpy(data[:, -1]).float()
            dataset = torch.utils.data.TensorDataset(features, labels)
            data_iter1 = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

            # 模型参数更新
            for x, y in data_iter1:
                # print(x.shape)
                out = model(x.to(device))  # 前向传播
                l = criterion(out, y.to(device).view(-1, 1))  # 计算损失
                optimizer.zero_grad()  # 梯度归零，避免累加
                l.backward()  # 反向传播
                optimizer.step()  # 参数更新
                print(l.item())
        except:
            pass
        finally:
            time.sleep(T)


def single_wait_features_vector(target_id):
    queue_filename = 'queue' + str(target_id) + '.txt'
    # print(queue_filename)
    with open(queue_filename, 'r') as f:
        line = f.readline()
    TR, IM = int(line.split(' ')[0]), int(line.split(' ')[1])
    xPU_file = open(str(target_id) + 'PUComputation.txt', 'r')
    xPU_lines = xPU_file.readlines()
    CPUi = int(xPU_lines[0].split(' ')[1])
    GPUi = int(xPU_lines[1].split(' ')[1])
    return torch.tensor([TR, IM, CPUi, GPUi]).float().to(device)


def wait_vectors():
    vector0 = single_wait_features_vector(0)
    vector1 = single_wait_features_vector(1)
    vector2 = single_wait_features_vector(2)
    vector3 = single_wait_features_vector(3)
    return [model(vector0).detach().numpy()[0], model(vector1).detach().numpy()[0],
            model(vector2).detach().numpy()[0], model(vector3).detach().numpy()[0]]

def single_sk_vector(target_id):
    queue_filename = 'queue' + str(target_id) + '.txt'
    # print(queue_filename)
    with open(queue_filename, 'r') as f:
        line = f.readline()
    TR, IM = int(line.split(' ')[0]), int(line.split(' ')[1])
    xPU_file = open(str(target_id) + 'PUComputation.txt', 'r')
    xPU_lines = xPU_file.readlines()
    CPUi = int(xPU_lines[0].split(' ')[1])
    GPUi = int(xPU_lines[1].split(' ')[1])
    return np.array([TR, IM, CPUi, GPUi])

def sk_vectors():
    vector0 = single_sk_vector(0)
    vector1 = single_sk_vector(1)
    vector2 = single_sk_vector(2)
    vector3 = single_sk_vector(3)
    vectors = [vector0,vector1,vector2,vector3]
    return regression.predict(vectors)

def wait_vectors_torch():
    vector0 = single_wait_features_vector(0)
    vector1 = single_wait_features_vector(1)
    vector2 = single_wait_features_vector(2)
    vector3 = single_wait_features_vector(3)
    return [model(vector0), model(vector1), model(vector2), model(vector3)]


# 读取now_execution设备的TR、im、CPUi and GPUi
def getCPUiandGPUi(now_execution):
    xPU_file = open(str(now_execution) + 'PUComputation.txt', 'r')
    xPU_lines = xPU_file.readlines()
    CPUi = int(xPU_lines[0].split(' ')[1])
    GPUi = int(xPU_lines[1].split(' ')[1])
    return CPUi, GPUi

def init_model():
    # 定义线性回归模型
    # torch.set_default_tensor_type(torch.DoubleTensor)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model = LinearRegression().to(device)
    # 定义损失函数MSELoss, 优化器SGD
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    data = [[0,0,1,2,0],
            [1,2,2,4,1],
            [1,2,1,2,2],
            [0,0,2,4,0],
            [3,6,1,2,6]]
    data = np.array(data)
    features = torch.from_numpy(data[:, :-1]).float()
    features[:, 2:4] = 1.0 / features[:, 2:4]
    labels = torch.from_numpy(data[:, -1]).float()
    dataset = torch.utils.data.TensorDataset(features, labels)
    data_iter1 = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    # 模型参数更新
    for x, y in data_iter1:
        # print(x.shape)
        out = model(x.to(device))  # 前向传播
        l = criterion(out, y.to(device).view(-1, 1))  # 计算损失
        optimizer.zero_grad()  # 梯度归零，避免累加
        l.backward()  # 反向传播
        optimizer.step()  # 参数更新
        # print(l.item())
    return model, criterion, optimizer, device

def init_sk_model():
    model = linear_model.LinearRegression()
    data = [[0, 0, 1, 2, 0],
            [1, 2, 2, 4, 1],
            [1, 2, 1, 2, 2],
            [0, 0, 2, 4, 0],
            [3, 6, 1, 2, 6]]
    data = np.array(data)
    x_train = data[:, :-1]
    x_train[:, 2:4] = 1.0 / x_train[:, 2:4]
    y_train = data[:, -1]
    model.fit(x_train, y_train)
    return model

def update_parameters_sk(model):
    # 时间周期T
    T = 10
    while True:
        # 读取数据
        data = union_TWL()
        if len(data) == 0:
            return
        data = np.array(data)
        x_train = data[:, :-1]
        x_train[:, 2:4] = 1.0 / x_train[:, 2:4]
        y_train = data[:, -1]
        model.fit(x_train, y_train)
        time.sleep(T)

def write_TWLstart(now_task,TR,IM):
    with open('TWLstart.txt', 'a') as f:
        w1 = str(now_task) + ' ' + str(TR) + ' ' + str(IM) + ' ' + str(time.time()) + '\n'
        f.write(w1)

def write_TWLend(now_task,TR,IM):
    # 模拟返回endtime
    with open('TWLend.txt', 'a') as f:
        w1 = str(now_task) + ' ' + str(time.time() + TR * 2 + IM * 3) + '\n'
        f.write(w1)

def single_arrange(now_task,executor):
    # 写任务安排
    arrangement[now_task] = executor
    with open('arrangement.txt', 'a') as f:
        f.write(str(now_task) + ' ' + str(executor) + '\n')
    # 读queue_executor.txt文件
    queue_txt_path = 'queue' + str(executor) + '.txt'
    with open(queue_txt_path,'r') as f:
        line = f.readline()
        if line == '':
            return
        TR = int(line.split(' ')[0])
        IM = int(line.split(' ')[1])
        # 写timestamp TR IM start_time
        write_TWLstart(now_task,TR,IM)
        write_TWLend(now_task,TR,IM)

    # 更新queue文件
    with open(queue_txt_path, 'w') as f:
        TR += 1
        IM += 2
        f.write(str(TR) + ' ' + str(IM))

def union_TWL():
    tasks_to_endtime = {}
    TWLs = []
    with open('TWLend.txt','r') as f:
        ends_lines = f.readlines()
        for line in ends_lines:
            task = line.split(' ')[0]
            if task not in arrangement.keys():
                print('union_TWL sleep')
                time.sleep(10)
                return TWLs
            endtime = line.split(' ')[1]
            tasks_to_endtime[task] = endtime
    with open('TWLstart.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            task = line.split(' ')[0]
            if task in tasks_to_endtime.keys():
                # TR, IM
                TR1 = line.split(' ')[1]
                IM1 = line.split(' ')[2]
                CPUi1,GPUi1 = getCPUiandGPUi(int(arrangement[int(task)]))
                twl = [float(TR1),float(IM1),float(CPUi1),float(GPUi1),float(float(tasks_to_endtime[task]) - float(line.split(' ')[3]))]
                TWLs.append(twl)
    return TWLs

now_times_already_allocation = []           # 已经分配的任务

# 得到所有任务
def merge_nowtime():
    now_times = []
    for i in range(4):
        filepath = 'nowtimes' + str(i+1) + '.txt'
        with open(filepath,'r') as f :
            lines = f.readlines()
            for line in lines:
                now_times.append(int(line))
    return now_times


# timestamp_to_arrange用来存储待分配的任务
timestamp_to_arrange = queue.Queue(maxsize=100)
timestamp_to_TR_IM = {}

# 任务分配方案字典, 存储每个任务的timestamp ：executor
arrangement = {}
# 读取toAllocate.txt文件, 返回所有待分配的任务
toAllocate_path = 'toAllocate.txt'
def read_to_allocate_to_queue():
    with open(toAllocate_path,'r') as f:
        timestamps = f.readlines()
        for timestamp in timestamps:
            if timestamp.endswith('\n'):
                timestamp = timestamp[:-1]
            if timestamp in arrangement.keys():
                continue
            timestamp_to_arrange.put(timestamp)

def task_arrangement():
    global flag
    # 时间周期T
    T = 20
    # 传输时延
    '''
    传送给第i执行者的传输成本TMi=MAXj(TMij)
    传输是并行的, 因此选取最大的TMij作为TMi
    TMij表示第i, j两个执行者之间的传输延迟
    '''
    TM = np.zeros((4, 4))
    while True:
        # 将所有任务合并，并筛选出未分配的任务
        # add_now_times_to_arrange()
        read_to_allocate_to_queue()
        while timestamp_to_arrange.empty() == False:
            # 更新参数
            Delay_1_2 = read_flie("Delay_1_2.txt")
            TM[1][2] = TM[2][1] = Delay_1_2
            Delay_1_3 = read_flie("Delay_1_3.txt")
            TM[1][3] = TM[3][1] = Delay_1_3
            Delay_2_3 = read_flie("Delay_2_3.txt")
            TM[2][3] = TM[3][2] = Delay_2_3
            # Delay_1_cloud = read_flie("Delay_1_cloud.txt")
            # TM[1][0] = TM[0][1] = Delay_1_cloud
            # Delay_2_cloud = read_flie("Delay_2_cloud.txt")
            # TM[2][0] = TM[0][2] = Delay_2_cloud
            # Delay_3_cloud = read_flie("Delay_3_cloud.txt")
            # TM[3][0] = TM[0][3] = Delay_3_cloud

            # 比较成本
            # 求出所有的cost
            # wait_matrix = wait_vectors()            # torch版
            wait_matrix = sk_vectors()
            # print(wait_matrix)
            Delay_matirx = np.max(TM[:, 1:], axis=1)
            #Delay_matirx = np.array([0,0,0,0])
            cost_matrix = Delay_matirx + wait_matrix
            cost_matrix[0] = float("inf")
            cost_matrix[3] = float('inf')

            # 计算出最小的cost
            cost_min = np.min(cost_matrix)
            # 根据最小cost的来源决定now_execution
            if (cost_min == cost_matrix[0]):
                executor = 0
            elif (cost_min == cost_matrix[1]):
                executor = 1
            elif (cost_min == cost_matrix[2]):
                executor = 2
            else:
                executor = 3
            now_task = timestamp_to_arrange.get()  # 得到队首任务编号
            # if flag:
            #     executor = 1
            # else:
            #     executor = 2
            # flag = not flag
            single_arrange(now_task,executor)

            send_arrangement()
            # send_file("deadline.txt")
            # send_file("now_execution.txt")
        time.sleep(T)

flag = True

if __name__ == '__main__':
    '''
    第i执行者执行队列预期等待时间：TWLi=f（TR，IN，CPU算力，GPU算力）
    f（TR，IN，CPU算力，GPU算力）使用机器学习回归模型求解
        TRi第i执行者表示目前等待执行的任务数
        INi第i执行者表示目前等待执行的图片数
        CPUi第i执行者的CPU算力
        GPUi第i执行者的GPU算力
    '''
    # 超参数
    epoch = 10
    batch_size = 1

    regression = init_sk_model()

    model, criterion, optimizer, device = init_model()
    init()
    threading.Thread(target=task_arrangement)
    #threading.Thread(target=update_parameters, args=(model, criterion, optimizer, device)).start()
    # threading.Thread(target=update_parameters_sk,args=(regression,)).start()
    task_arrangement()
    # update_parameters(model, criterion, optimizer, device)



def send_file(filename):
    for i in range(drone_num):
        send_file_path = '/home/tzy/sjj/TD_cloud+3/cloud/taskAllocate/' + filename
        destination_1 = "/home/tzy/sjj/TD_cloud+3/drone_"+str(i+1)+"/taskAllocate/"
        order = "python3 " + send_py_path + " "+ str(i+1)+" " + send_file_path + " " + destination_1
        print(order)
        os.system(order)
    # 发1号无人机
    # destination_1='linpeng@10.10.114.14:/home/linpeng/new/TD_cloud+3/drone_1'
    # order="scp "+filename+" " +destination_1
    # os.system(order)
    # send_file_path = '/home/tzy/sjj/TD_cloud+3/cloud/任务分配/' + filename
    # destination_1 = "/home/tzy/sjj/TD_cloud+3/drone_1/任务分配"
    # order = "python3 "+send_py_path+" 1 " + send_file_path + " " + destination_1
    # os.system(order)

    # 发2号无人机
    # destination_2 = "linpeng@10.10.114.14:/home/linpeng/new/TD_cloud+3/drone_2"
    # order = "scp " + filename+" " + destination_2
    # os.system(order)
    # send_file_path = '/home/tzy/sjj/TD_cloud+3/cloud/任务分配/' + filename
    # destination_2 = "/home/tzy/sjj/TD_cloud+3/drone_2/任务分配"
    # order = "python3 "+send_py_path+" 2 " + send_file_path + " " + destination_1
    # os.system(order)

    # 发3号无人机
    # destination_3 = "linpeng@10.10.114.14:/home/linpeng/new/TD_cloud+3/drone_3"
    # order = "scp " + filename + " " + destination_3
    # os.system(order)
    # send_file_path = '/home/tzy/sjj/TD_cloud+3/cloud/任务分配/' + filename
    # destination_3 = "/home/linpeng/new/unknown_pictures/TD_cloud+3/drone_3"
    # order = "python3 "+send_py_path+" 3 " + send_file_path + " " + destination_1
    # os.system(order)

