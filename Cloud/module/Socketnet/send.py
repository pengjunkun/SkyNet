#!/usr/bin/python3
# -*- coding: utf-8 -*-

import socket
import os
import hashlib
import sys
import threading
import time
import ipconfig

BUFFERSIZE = 4096
SEPARATOR = "<SEPARATOR>"
'''
文件夹传输客户端程序
传输协议：
1. 基于TCP协议；
2. 客户端连接服务器成功后，客户端不发送任何消息，服务器端首先将文件的描述信息
    （定长包头，长度为347B）发送给客户端，紧接着发送文件数据给客户端，
    发送文件数据后断开连接；
3. 文件描述信息结构为：文件相对发送文件夹的相对路径名（300B，右边填充空格，UTF-8编码）
    +文件大小（15B，右边填充空格）+ 文件MD5值（32B, 大写形式）
4. 若文件夹中有空文件夹，发送空文件夹时，文件描述信息为：空文件夹相对发送文件夹的相对路径名
    （300B，右边填充空格，UTF-8编码）
    +文件大小设为-1（15B，右边填充空格）+ 文件MD5值设为(' ' *32),即32个空格（32B, 大写形式）
'''


def get_file_md5(file_name_path):
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


def recv_file(sock):
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
            print("成功接收空文件夹！{}".format(file_name_path))
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
        os.makedirs(os.path.dirname(file_name_path), exist_ok=True)

        file_name = os.path.basename(file_name_path)
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
            print(file_name)
            print(str(recv_size))
            print("\r正在接收{}文件，已接收{}字节".format(file_name, recv_size), end=' ')
        f.close()

        end_time = time.time()
        t = end_time - start_time
        # 调用MD5生成函数检验收到的文件是否是发送方的原文件
        recv_file_md5 = get_file_md5(file_name_path)
        if recv_file_md5 == file_md5:
            print("\n成功接收文件{},用时{:.2f}s\n".format(file_name, t))
        else:
            print("接收文件 %s 失败（MD5校验不通过）\n" % file_name)
            break

def send_empty_dir(sock_conn, file_abs_path, dest_file_parent_path):
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


def send_one_file(sock_conn, file_abs_path, dest_file_parent_path):
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
    file_md5 = get_file_md5(file_abs_path)
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


def send_file_thread(sock_conn, dest_file_abs_path, dest_file_parent_path):
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
                send_empty_dir(sock_conn, root, dest_file_parent_path)
            # 如果有文件，就调用发送文件的函数
            for f in files:
                file_abs_path = os.path.join(root, f)  # 获取文件的绝对路径
                print(file_abs_path)
                send_one_file(sock_conn, file_abs_path, dest_file_parent_path)
    except Exception as e:
        print(e)
    finally:
        sock_conn.close()


def send_only_one_file(sock_conn,file_abs_path,dest_file_parent_path):
    file_name = file_abs_path[len(dest_file_parent_path):]  # 获取目标文件夹的名字
    # 切完后的file_name可能是这样的“\a”,在Linux下“/a”，去掉斜杠
    if file_name[0] == '\\' or file_name[0] == '/':
        file_name = file_name[1:]

    file_size = os.path.getsize(file_abs_path)
    file_md5 = get_file_md5(file_abs_path)
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

def main():
    # send_id = input('请输入目标主机的id号')
    # send_file_path = input('请输入本地传输文件的绝对路径')
    # dest_file_path = input('请输入目的主机的文件路径')

    send_id = sys.argv[1]
    send_file_path = sys.argv[2]
    dest_file_path = sys.argv[3]

    send_file_parent_path = os.path.dirname(send_file_path)  # 获取目标文件夹的父目录的绝对路径

    server_ip = ipconfig.id_ip[send_id]
    # print(server_ip)
    server_port = 6789

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
        threading.Thread(target=send_file_thread, args=(sock, send_file_path, send_file_parent_path)).start()
    else:
        threading.Thread(target=send_only_one_file, args=(sock, send_file_path, send_file_parent_path)).start()

    # sock.close()


if __name__ == "__main__":
    main()
