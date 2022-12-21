# SkyNet Drone Side
## Overview
This is the drone device in the SkyNet system.
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
	Pillow
	joblib
	sklearn
	pandas
	insightface
	onnx
	onnxruntime-gpu

## Document description
	“delay”：The transfer delay obtained by sending empty files
	“detect_result”：Save the detected result
	“exp_pool”：Data for linear model updates
	“final_result”：Final result
	"information"：Device information
	“module”：System module of the main code
	“start”：The place where put the signal that make the multi-device synchronization starts
	“task_to_allocate”：Task assignment file
	“tasks”：Pending task
	“Cloud.py”：mian code to call each module


## Ready for use
- Before using the system, it is recommended to test each module of the system. For specific test methods, refer to reademe in each module.
- Deploy the system on at least three devices, one cloud and two drones.
- Modify the location of the main folder in the main function.
- Change the device IP address in the communication module (socketnet).
- Put the data in the right place in each drone side: e.g., ./Drone1/task/bb_1/1_1.jpg, ./Drone2/task/bb_2/2_1.jpg.
- Start the system，first start each drone，then start the cloud：
```
python Drone.py
python cloud.py
```

## Notice
- Multi-device jobs are difficult to build. If there are no multiple devices, it is suggested to test our module, which is easier to use.
- The translation of code comments is relatively large, and the comments of Cloud.py have been translated, Drone.py's code structure is similar to cloud-py, so you can refer to the comments of Cloud.py. 