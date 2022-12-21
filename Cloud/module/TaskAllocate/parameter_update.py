import os
import time

yhj = True

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
# TD_path = 'D:/DevProjects/python/MCOT_nano/sjj/TD_cloud+3'

cloud_allocate_dir_path = TD_path_cloud + 'cloud/taskAllocate/'

# 此无人机编号
drone_id = '1'
# 图片文件参数设置
img_dir = TD_paths[drone_id] + 'drone_' + drone_id + '/ObjectDetection+featuresRec/img/'
to_allocate_path = 'toAllocate.txt'  # 记录存有所有待分配文件的时间戳的文件
arrangement_path = 'arrangement.txt'

# 本机网络传输文件路径配置
send_py_path = TD_paths[drone_id] + 'drone_' + drone_id + '/net/send.py'


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


def send_file(filename):
    # 发云端
    # destination='linpeng@10.10.114.14:/home/linpeng/new/TD_cloud+3/cloud'
    # order="scp "+filename +" "+destination
    # os.system(order)
    send_file_path = TD_paths[drone_id] + 'drone_' + drone_id + '/taskAllocate/' + filename
    destination_0 = cloud_allocate_dir_path
    order = "python3 " + send_py_path + " 0 " + send_file_path + " " + destination_0
    os.system(order)


def select_to_allocate():
    # 待分配的任务时间戳
    to_allocate = []
    # print(img_dir)
    img_paths = os.listdir(img_dir)
    for img_path in img_paths:
        # 截取时间戳，start_index为_的下标的下一位，end_index为最后一个.的下标
        start_index = img_path.find('_') + 1
        end_index = img_path.rfind('.')
        timestamp = img_path[start_index:end_index]
        # 如果此时间戳的任务没有被分配，则加入到待分配中
        if timestamp not in already_allocate and timestamp not in to_allocate:
            to_allocate.append(timestamp)
            already_allocate.append(timestamp)
    return to_allocate


def write_to_allocate(to_allocate):
    to_allocate_str = ''
    with open(to_allocate_path, 'w') as f:
        # 遍历每个待分配的时间戳文件
        for timestamp in to_allocate:
            to_allocate_str += timestamp + '\n'
        f.write(to_allocate_str)


def send_to_allocate_to_cloud():
    # 云端目标文件的路径
    target_path = cloud_allocate_dir_path
    # toAllocate.txt文件的绝对路径
    send_file_path = os.path.abspath(to_allocate_path)
    order = 'python3 ' + send_py_path + ' 0 ' + send_file_path + ' ' + target_path
    print(order)
    os.system(order)


# 读取arrangement.txt文件，返回timestamp : executor 字典
def read_arrangement():
    arrangement_dict = {}
    with open(arrangement_path, 'r') as f:
        pairs = f.readlines()
        for timestamp_executor in pairs:
            timestamp = timestamp_executor.split(' ')[0]
            executor = timestamp_executor.split(' ')[1]
            if executor.endswith('\n'):
                executor = executor[:-1]
            arrangement_dict[timestamp] = executor
    return arrangement_dict


# 将时间戳文件发送给目的机器
def send_img(timestamp, executor):
    # 获取文件路径
    if drone_id != str(executor):
        # 本机bb_i_j路径
        bb_path = TD_paths[drone_id] + 'drone_' + str(drone_id) + '/ObjectDetection+featuresRec/img_out/bb_' + str(
            drone_id) + '/bb_' + str(
            drone_id) + '_' + str(timestamp)
        send_file_path = bb_path
        while not os.path.exists(send_file_path):
            time.sleep(1)
        # 目标文件夹路径
        # target_path = 'D:\\DevProjects\\python\\MCOT_nano\\sjj\\TD_cloud+3\\drone_'+ executor +'\\img\\'
        target_path = TD_paths[str(executor)] + 'drone_' + str(executor) + '/ObjectDetection+featuresRec/img_out/bb_' + str(
            drone_id) + '/'
        order = 'python3 ' + send_py_path + ' ' + executor + ' ' + send_file_path + ' ' + target_path
        print(order)
        os.system(order)
        order = 'rm -r ' + bb_path
        os.system(order)


# 按照arrangement.txt将图片分给对应的executor
def arrange():
    arrangement_dict = read_arrangement()
    for timestamp in arrangement_dict.keys():
        executor = arrangement_dict[timestamp]
        send_img(timestamp, executor)


# 分配周期设置T，每T秒扫描一次文件
T = 1000

already_allocate = []  # 已经分配的任务的时间戳集合


def allocate():
    # 获取还没有被分配的时间戳的图片
    to_allocate = select_to_allocate()
    print(to_allocate)
    # 将to_allocate写入到文件中
    write_to_allocate(to_allocate)
    # 将to_allocate_path文件发送到云端
    send_to_allocate_to_cloud()
    # 读取arrangement.txt文件, 将图片分发给目的端
    arrange()


while (True):
    # estimated_time = read_flie("estimated_time.txt")
    # queue_length = read_flie("queue_length.txt")
    # wait_1 = estimated_time * queue_length
    # write_flie("wait_1.txt", wait_1)
    # send_file("wait_1.txt")

    now_time = time.time()
    write_flie("Delay_1_cloud.txt", now_time)
    send_file("Delay_1_cloud.txt")

    Delay_1_2 = read_flie("Delay_1_2.txt")
    # print("Delay_1_2:" + str(Delay_1_2))

    a = os.path.getmtime("Delay_1_2.txt")
    # print("getmtime:Delay_1_2.txt:" + str(a))
    Delay_1_2 = a - Delay_1_2
    write_flie("Delay_1_2.txt", Delay_1_2)
    send_file("Delay_1_2.txt")

    # Delay_1_3 = read_flie("Delay_1_3.txt")
    # b = os.path.getmtime("Delay_1_3.txt")
    # Delay_1_3 = b - Delay_1_3
    # write_flie("Delay_1_3.txt", Delay_1_3)
    # send_file("Delay_1_3.txt")

    # print("wait_1:"+str(wait_1))
    # print("Delay_1_2:" + str(Delay_1_2))
    # print("Delay_1_3:" + str(Delay_1_3))
    # print("Delay_1_cloud:" + str(now_time))
    # print('--------------------')

    # 读取时间戳，选择出未分配的任务并发送给云端
    allocate()
    time.sleep(T)
