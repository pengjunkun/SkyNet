# SkyNet

## Overview
Multi-Drone Cooperation for Real-Time Person Identification and Localization. 

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
	“Cloud”：The cloud side in skynet
	“Drone”：one drone in skynet


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
Multi-device jobs are difficult to build. If there are no multiple devices, it is suggested to test our module, which is easier to use. The readme of each module are in：
- ./Cloud(Drone)/module/Alignment/README - en.md
- ./Cloud(Drone)/module/Detection/README - en.md
- ./Cloud(Drone)/module/Identification/README - en.md

## Contacts
If you have any problem about this paper/repository or want to discuss the topic with us, you are welcomed to send emails to pengjunkun@gmail.com

Thank you!
