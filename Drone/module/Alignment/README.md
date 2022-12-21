# 对齐算法
## 算法概况
  假设接收多架无人机传输的帧，按指定格式存储（与粗检测的存储格式相同），按格式读取，并使用字典实现时间戳对齐，对每一时间戳进行对齐，使用三维点遍历投影视图，将同一物体的框图放于同一个文件夹中。

## 环境要求
	opencv-python
	numpy
	Pillow
	tqdm

## 文件
	alignment.py：对齐算法主函数；
	getpixel.py：手动获得图像二维坐标点；
	align_bb_0/bb_i：第i个无人机传入的帧与检测的boundingbox，格式与粗检测相同，子文件夹后缀为时间戳；
	align_bb_0/get3D：对齐后的图像存储，按照时间戳/物体编号/图片的路径存储；
	3/2Dpoint.txt：特征点的三维与二维坐标

## 使用方法
（1）编辑好特征点坐标至txt文件中；
（2）将多无人机传入的数据按格式分好：align_bb_0/bb_i
（3）运行代码：python alignment.py --time=当前时间戳
例：

	python alignment.py --time=1

## 输出结果
在文件夹align_bb_0/get3D将会得到对齐的结果，子文件夹为时间戳，子子文件夹objecti为第i个对象在不同无人机下检测出的视图。