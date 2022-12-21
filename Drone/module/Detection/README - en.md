# Object detection + feature point detection algorithm
## Overview
YOLOx was used to realize single frame image detection, simulate multi-UAV incoming multi-frame detection, call the camera real-time detection, and retrain the network to realize parallel feature point detection.

## Required environment
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

## Document description
	“img”：Image detection of the original image storage folder
	“img_out”：Output folder
	“logs”：log output for training
	“model_data”：Model parameter
	"nets"：Network backbone implementation
	“tools”：Video processing tool
	“utils”：Data analysis tool
	“VOCdevkit”：Address for storing VOC data sets
	“predict.py”：Prediction code (mainly runs this)
	“train.py”：Training code
	“yolo.py”：yolox network creation
	“yolo_barricade.py”：The feature point detection network is created
Other specific files can be found in the nets folder, YOLOX-README。

## Ready for use
- Image detection: The image to be detected is stored in the img folder. The corresponding mode is "predict", "dir_predict".
- Video check: Save videos in the source file directory. The corresponding modes are video or video_deep_sort.

## Usage pattern
- “predict”：Enter the name of a single image and check. The image is stored in img
- “video”：Call the camera for real-time detection. According to the requirements of data format, keep the detection frame and picture of each frame to the bb_i folder (in img_out) and save the video to the root directory
- “fps”：Test detection speed
- “dir_predict”：Detect all images in img folder, which is used to simulate the coarse detection of multiple drones
- “predict_single”：Single-view detection algorithm that saves boundingbox without alignment.

## Usage method
python predict.py 
- --mode=The corresponding name of the schema 
- --UAV_ID=Number of UAV 
- --camera=Camera number
- --IMG=Image address

example：
（1）Single image detection mode：

	python predict.py --mode=predict --UAV_ID=2 --IMG=2_1.jpg

（2）Multi-picture detection simulates the coarse detection mode of multi-UAV (note that UAV_ID must be 1, because in simulates multi-UAV, this ID is only used for counting)：

	python predict.py --mode=dir_predict --UAV_ID=1

（3）Call the camera for continuous detection (in the mode of real operation on the UAV)：

	python predict.py --mode=video --UAV_ID=1 --camera=0

（4）Test detection speed：

	python predict.py --mode=fps

（5）Single view image detection mode, directly generate each object：

	python predict.py --mode=predict_single --UAV_ID=1 --IMG=1_1.jpg

## Output result
View the results in img_out, where
- If the analog multi-view mode "dir_predict" is used, align_bb_0 is generated: it simulates the crude detection of multiple drones and then transmits it to the host, where bb_i represents the detection result of each drone. The subfolder suffix is time stamp, and each subfolder contains the txt file of frames and detected frames.
- If the UAV image detection mode "predict" is used, bb_i: the result of real-time detection by the current UAV i will be generated. The subfolder format is the same as above.
- If the single-view detection mode "predict_single" is used, folder i will be generated, where i represents the timestamp of the detected image, and interior objectj, where j represents the JTH object detected.

