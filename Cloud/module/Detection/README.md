# 目标检测+特征点检测算法
## 算法概况
使用YOLOx实现了单帧图像检测，模拟多无人机传入多帧检测，调用摄像头实时检测，同时重训练网络，实现并行的特征点检测。

## 所需环境
	scikit-learn
	scipy == 1.4.1
	numpy == 1.17.0
	matplotlib == 3.1.2
	opencv_python == 4.1.2.30
	torch >= 1.3.0
	torchvision >= 0.4.1
	tqdm == 4.60.0
	Pillow == 8.2.0
	h5py == 2.10.0

## 文件说明
	“img”：图像检测的原图存储文件夹
	“img_out”：输出文件夹
	“logs”：训练的log输出
	“model_data”：模型参数
	"nets"：网络骨干实现
	“tools”：视频处理工具
	“utils”：数据分析工具
	“VOCdevkit”：VOC数据集存放地址
	“predict.py”：预测代码（主要运行这个）
	“train.py”：训练代码
	“yolo.py”：yolox网络创建
	“yolo_barricade.py”：特征点检测网络创建
其他的具体文件可见YOLOX-README，放于nets文件夹中。

## 使用准备
- 图片检测：在img文件夹中存放待检测的图片，对应模式为“predict”，“dir_predict”，
- 视频检测，在源文件目录下存放视频，对应模式为“video”，“video_deep_sort”

## 使用模式
- “predict”：输入单张图片名字后检测，图片存放在img
- “video”：调用摄像头进行实时检测，按照数据格式要求，保留每一帧的检测框与图片到bb_i文件夹（在img_out中），保存视频存放在根目录
- “fps”：测试检测速度
- “dir_predict”：检测img文件夹所有的图片，用来模拟多个无人机的粗检测
- “predict_single”：单视角的检测算法，直接保存boundingbox而不用对齐。

## 使用方法
python predict.py 
- --mode=模式的对应名称 
- --UAV_ID=无人机的编号 
- --camera=摄像头的编号 
- --IMG=图像的地址

例子：
（1）单张图片检测模式：

	python predict.py --mode=predict --UAV_ID=2 --IMG=2_1.jpg

（2）多张图片检测模拟多无人机粗检测模式（注意UAV_ID一定要为1，因为模拟多无人机，该ID只用来计数）：

	python predict.py --mode=dir_predict --UAV_ID=1

（3）调用摄像头连续检测（在无人机上实机运行的模式）：

	python predict.py --mode=video --UAV_ID=1 --camera=0

（4）测试检测速度：

	python predict.py --mode=fps

（5）单视角图片检测模式，直接生成各object：

	python predict.py --mode=predict_single --UAV_ID=1 --IMG=1_1.jpg

## 输出结果
在img_out中查看结果，其中
- 若使用模拟多视角模式“dir_predict”，会生成align_bb_0：模拟多个无人机粗检测之后传到主机，其中bb_i代表每一个无人机的检测结果，子文件夹后缀为时间戳，每一个子文件夹中包括帧与检测得到的框的txt文件；
- 若使用无人机图片检测模式“predict”，会生成bb_i：当前无人机i实时检测得到的结果，子文件夹格式与上述相同。
- 若使用单视角检测模式“predict_single”，会生成文件夹i，i代表检测图片的时间戳，内部为objectj，j代表检测到的第j个物体。

