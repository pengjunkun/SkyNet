import socket
import os
import hashlib
import sys
import threading
import time
#import ipconfig
#import fcntl

class Socketnet(object):

    def __init__(self,device_id):
        self.device_id=device_id

        self.id_ip = {
        '0': '192.168.189.24',
        '1': '192.168.189.111',
        '2': '192.168.189.231',
        '3': '127.0.0.1'
        }
        '''

        self.id_ip = {
        '0': '192.168.189.24',
        '1': '192.168.189.24',
        '2': '192.168.189.24',
        '3': '127.0.0.1'
        }
        '''

        self.id_name = {
        '0':'云端',
        '1':'无人机-1',
        '2':'无人机-2',
        '3':'无人机-3',
        }

        self.id_port={
            '0':6789,
            '1':6790,
            '2':6791,
            '3':6792,
        }

        self.ip_id = {}
        for key, val in self.id_ip.items():
            self.ip_id[val] = key
        self.name_id = {}
        for key, val in self.id_name.items():
            self.name_id[val] = key

        self.recv_flag=0

    def get_file_md5(self,file_name_path):
        '''
        函数功能：将发送过来的文件生成MD5值
        参数描述：
            file_name_path 接收文件的绝对路径

        '''
        m = hashlib.md5()
        # print(file_name_path)
        with open(file_name_path, "rb") as f:
            while True:
                data = f.read(1024)
                if len(data) == 0:
                    break
                m.update(data)

        return m.hexdigest().upper()


    def recv_file(self,sock):
        '''
        函数功能：根据协议循环接收包头，根据接收到的包头（文件描述信息结构）创建文件夹及其子文件夹，
                 在相应目录下写入文件
        参数描述：
            sock 连接套接字

        '''
        while True:
            # 根据协议先接收文件名或空文件夹名的相对路径的字节
            file_name_path = sock.recv(300).decode().rstrip()
            print(file_name_path)
            # 若文件名长度为0，说明接收文件夹完毕，跳出循环
            if len(file_name_path) == 0:
                break
            # 接收文件大小的字节
            file_size = sock.recv(15).decode().rstrip()
            # 若文件大小字节为'-1'，根据协议接收到的为空文件夹
            if int(file_size) == -1:
                #print("成功接收空文件夹！{}".format(file_name_path))
                os.makedirs(file_name_path, exist_ok=True)
                continue  # 继续循环
            # 若接收到的文件大小字节的长度为0，说明接收文件夹完毕，跳出循环
            if len(file_size) == 0:
                break
            # print(file_size)
            # 接收MD5字节，若长度为0，跳出循环
            file_md5 = sock.recv(32).decode().rstrip()
            if len(file_md5) == 0:
                break
            # print(file_md5)
            # 根据接收到的包头信息，创建文件夹
            try:
                os.makedirs(os.path.dirname(file_name_path), exist_ok=True)

                file_name = os.path.basename(file_name_path)
            except:
                file_name=file_name_path
            # 根据文件大小循环接收并写入
            recv_size = 0
            start_time = time.time()

            f = open(file_name_path, "wb")  # 初始化已接收到的字节
            while recv_size < int(file_size):  # 若接收到的字节小于文件大小就循环接收，并写入本地
                file_data = sock.recv(int(file_size) - recv_size)
                if len(file_data) == 0:
                    break

                f.write(file_data)

                recv_size += len(file_data)
                #print(file_name)
                #print(str(recv_size))
                #print("\r正在接收{}文件，已接收{}字节".format(file_name, recv_size), end=' ')
            f.close()

            end_time = time.time()
            t = end_time - start_time
            # 调用MD5生成函数检验收到的文件是否是发送方的原文件
            recv_file_md5 = self.get_file_md5(file_name_path)
            if recv_file_md5 == file_md5:
                #print("\n成功接收文件{},用时{:.2f}s\n".format(file_name, t))
                1
            else:
                #print("接收文件 %s 失败（MD5校验不通过）\n" % file_name)
                break

    def send_empty_dir(self,sock_conn, file_abs_path, dest_file_parent_path):
        '''
        函数功能：将目标文件夹中所含的空文件夹发送给客户端
        参数描述：
            sock_conn 套接字对象
            dir_abs_path 空文件夹文件的绝对路径
            dest_file_parent_path 待发送文件的文件目录的相对路径（相对于发送的文件夹）
        '''
        file_name = file_abs_path[len(dest_file_parent_path):]  # 获取目标文件夹的名字
        # 切完后的file_name可能是这样的“\a”,在Linux下“/a”，去掉斜杠
        if file_name[0] == '\\' or file_name[0] == '/':
            file_name = file_name[1:]
        # 将空文件夹下的文件的大小设为-1，MD5设为32个空格
        file_size = -1
        file_md5 = ' ' * 32
        # 根据传输协议，设置包头内容
        file_name = file_name.encode()
        file_name += b' ' * (300 - len(file_name))
        file_size = "{:<15}".format(file_size).encode()

        file_desc_info = file_name + file_size + file_md5.encode()  # 加码后的包头内容
        # 发送空文件夹的包头
        sock_conn.send(file_desc_info)


    def send_one_file(self,sock_conn, file_abs_path, dest_file_parent_path):
        '''
        函数功能：将一个文件发送给客户端
        参数描述：
            sock_conn 套接字对象
            file_abs_path 待发送的文件的绝对路径
            dest_file_parent_path 待发送文件的文件目录的相对路径（相对于发送的文件夹）
        '''
        file_name = file_abs_path[len(dest_file_parent_path):]  # 获取目标文件夹的名字
        # 切完后的file_name可能是这样的“\a”,在Linux下“/a”，去掉斜杠
        if file_name[0] == '\\' or file_name[0] == '/':
            file_name = file_name[1:]

        file_size = os.path.getsize(file_abs_path)
        file_md5 = self.get_file_md5(file_abs_path)
        # 将空文件夹下的文件的大小设为-1，MD5设为32个空格
        file_name = file_name.encode()
        file_name += b' ' * (300 - len(file_name))
        file_size = "{:<15}".format(file_size).encode()

        file_desc_info = file_name + file_size + file_md5.encode()  # 加码后的包头内容
        # 发送目标文件夹下发送文件的包头
        sock_conn.send(file_desc_info)
        # 变读取文件内容变发送
        with open(file_abs_path, "rb") as f:
            while True:
                data = f.read(1024)
                if len(data) == 0:
                    break
                sock_conn.send(data)


    def send_file_thread(self,sock_conn, dest_file_abs_path, dest_file_parent_path):
        '''
        函数功能：遍历目标文件夹下的子文件夹、以及所有文件，调用发送文件函数和发送空文件夹函数
        参数描述：sock_conn 连接套接字
                 dest_file_abs_path 目标文件夹的绝对路径
                 dest_file_parent_path 目标文件夹父目录的绝对路径
        '''
        try:
            # print(dest_file_parent_path)
            for root, dirs, files in os.walk(dest_file_abs_path):
                # 如果是空文件夹，就掉用空文件夹发送函数
                # print('True')
                if len(dirs) == 0 and len(files) == 0:
                    self.send_empty_dir(sock_conn, root, dest_file_parent_path)
                # 如果有文件，就调用发送文件的函数
                for f in files:
                    file_abs_path = os.path.join(root, f)  # 获取文件的绝对路径
                    print(file_abs_path)
                    self.send_one_file(sock_conn, file_abs_path, dest_file_parent_path)
        except Exception as e:
            print(e)
        finally:
            sock_conn.close()


    def send_only_one_file(self,sock_conn,file_abs_path,dest_file_parent_path):
        file_name = file_abs_path[len(dest_file_parent_path):]  # 获取目标文件夹的名字
        # 切完后的file_name可能是这样的“\a”,在Linux下“/a”，去掉斜杠
        if file_name[0] == '\\' or file_name[0] == '/':
            file_name = file_name[1:]

        file_size = os.path.getsize(file_abs_path)
        file_md5 = self.get_file_md5(file_abs_path)
        # 将空文件夹下的文件的大小设为-1，MD5设为32个空格
        file_name = file_name.encode()
        file_name += b' ' * (300 - len(file_name))
        file_size = "{:<15}".format(file_size).encode()

        file_desc_info = file_name + file_size + file_md5.encode()  # 加码后的包头内容
        # 发送目标文件夹下发送文件的包头
        sock_conn.send(file_desc_info)
        # 变读取文件内容变发送
        with open(file_abs_path, "rb") as f:
            while True:
                data = f.read(1024)
                if len(data) == 0:
                    break
                sock_conn.send(data)

    def send(self,send_id,send_file_path,dest_file_path):

        send_file_parent_path = os.path.dirname(send_file_path)  # 获取目标文件夹的父目录的绝对路径

        server_ip = self.id_ip[send_id]
        # print(server_ip)
        server_port = self.id_port[send_id]

        ip_port = (server_ip, server_port)   # 对方的IP地址与端口号
        sock = socket.socket()               # 建立TCP socket
        sock.connect(ip_port)                # 与对方建立连接
        # print(sock._closed)

        # 300B
        dest_file_path = dest_file_path.encode()
        dest_file_path += b' ' * (300 - len(dest_file_path))
        sock.send(dest_file_path)

        # 调用发送文件进程
        if os.path.isdir(send_file_path):
            t=threading.Thread(target=self.send_file_thread, args=(sock, send_file_path, send_file_parent_path))

        else:
            t=threading.Thread(target=self.send_only_one_file, args=(sock, send_file_path, send_file_parent_path))
        t.start()
        #t.join()
        while True:
            #print(t.isAlive())
            if t.isAlive():
                continue
            else:
                sock.close()
                return




    def recv_thread(self,thisId):
        # 绑定套接字，监听
        # ip_port = ('192.168.57.129', 6789)
        ip = self.id_ip[thisId]
        port = self.id_port[thisId]
        ip_port = (ip, port)
        sock_listen = socket.socket()
        # 防止端口被占用
        sock_listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock_listen.bind(ip_port)
        sock_listen.listen(5)
        while True:
            print(str(self.id_name[thisId]) + '接收方监听中..... ')
            sock, addr = sock_listen.accept()
            dest_file_abs_path = sock.recv(300).decode().rstrip()
            dest_file_abs_path.strip()
            # print(dest_file_abs_path + ' 1')
            os.chdir(r'' + dest_file_abs_path)
            if 'tasks/bb_' in dest_file_abs_path:
                self.recv_flag=1
            if 'exp_poll' in dest_file_abs_path:
                self.recv_flag=2
            self.recv_file(sock)
            self.recv_flag=0

        sock_listen.close()
    def recv(self):
        t=threading.Thread(target=self.recv_thread,args=self.device_id)
        #t.setDaemon(True)
        t.start()




