# 检测+多路视图融合识别

## 算法简介
使用insightface对人脸进行检测与特征提取，采用加权平均对特征进行融合，最后使用SVM分类器进行特征的分类，得到最后的ID结果。

## 环境要求
- python 3.6
- pytorch >=1.0
- numpy >=1.13.1
- os
- Pillow
- joblib
- sklearn
- pandas
- insightface
- onnx
- onnxruntime-gpu（要安装与cuda版本匹配的版本）

## 文件说明
- crop_img：存储生成数据库时检测的人脸；
- img：数据库的图片；
- test：待测试的图片存放，子文件夹objecti为同一物体的图片集合，子文件夹下的才是图片；
- userdata：存放生成的数据库，包括人名与数据库图片的特征向量；同时存放了SVM分类器的训练结果，用户注册时的视频与图片结果。
- date_create.py：生成数据库的函数；
- main.py：人脸检测与识别的主函数
- Register.py：使用摄像头拍摄数据；
- SVM.py：训练SVM分类器函数；
- warm_input.jpg：用来给神经网络热身的漂亮女生照片/// _ ///
- test_result：存放测试图片中裁剪下来的人脸。

## 使用准备
（1）将需要识别的人的模板照片放于img文件夹中（注意不是待检测照片），同一人的照片放于同一文件夹下，必须命名为该人的ID；
（2）若是想要注册自己的数据，运行：

	python Register.py

此时会要求输入人的ID，之后将会调用摄像头，按照提示按a键开始录制10s的视频，录制的时候尽量多角度。之后将会储存在userdata/origin文件夹下。在文件夹中选取几张角度不同的图片，复制到img文件夹中即可。
（3）生成数据库：

	python data_create.py

（4）训练SVM分类器（20220302：不需要了，使用距离度量）：

	python SVM.py

若没有向数据库中添加新的人，则不需要进行上述四步的准备工作。直接开始第（5）步。

（5）将待检测的人的照片放于test文件夹下，同一人的照片放于同一文件夹下，可命名为object<i>。注意，若该人的类别不在数据库中，则不能被识别到，需要向数据库中注册该人的数据，即从（1）开始。（没有做unknown类是因为unknown的特征过多，SVM不好分类）。

## 使用方法
做好准备工作后，运行：

	python main.py --time=1 --fusion=arcfusion

其中time代表时间戳，fusion代表融合模式，有姿态融合：arcfusion、平均融合：averagefusion

## 结果查看
在控制台会实时输出检测图片的ID，同时在result.txt中保存了同一文件夹下图片对应的ID。

同时在控制台会输出检测的时间，取决于图片的分辨率。

