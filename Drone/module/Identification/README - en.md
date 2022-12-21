# Multi-channel view fusion identification

## Algorithm introduction
Insightface was used for face detection and feature extraction, weighted average was used for feature fusion, and finally SVM classifier was used for feature classification to obtain the final ID result.

## Environmental requirements
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
- onnxruntime-gpu（Install a version that matches the cuda version）

## Document description
- crop_img：Store the faces detected when the database is generated;
- img：A picture of the database;
- test：Images to be tested are stored. Subfolder objecti is a collection of images of the same object, and only images are in subfolders.
- userdata：Store the generated database, including the feature vector of names and database pictures; At the same time, the training results of SVM classifier, video and picture results of user registration are also stored.
- date_create.py：A function that generates a database;
- main.py：Face detection and recognition of the main function
- Register.py：Using cameras to capture data;
- SVM.py：Training SVM classifier function;
- warm_input.jpg：Photos of pretty girls used to warm up neural networks /// _ ///
- test_result：Store cropped faces from test images.

## Ready for use
（1）Place the template photo of the person to be identified in the img folder (note that it is not the photo to be detected). If the photo of the same person is placed in the same folder, it must be named as the ID of that person.
（2）If you want to register your data, run

	python Register.py

At this time, you will be asked to input the ID of the person. Then, the camera will be called and press a as prompted to record the video for 10 seconds from as many angles as possible. It will then be stored in the userdata/origin folder. Select a few images from different angles in the folder and copy them to the img folder.
（3）To generate the database:

	python data_create.py

（4）Training SVM classifier (20220302: no need, use distance measure) :

	python SVM.py

If you have not added a new person to the database, you do not need to perform the above four steps of preparation. Go straight to step (5).

（5）Store the photo of the person to be detected in the test folder and the photo of the same person in the same folder. The photo can be named object<i>. Note that if the person's category is not in the database, it is not recognized and the person's data needs to be registered with the database, i.e., starting with (1). (unknown class is not made because there are too many features of unknown, so SVM is not good for classification).

## Usage method
After making preparations, run:

	python main.py --time=1 --fusion=arcfusion

Where, time represents time stamp, fusion represents fusion mode, attitude fusion: arcfusion, averagefusion: averagefusion
## Result viewing
The console outputs the ID of the detected image in real time, and saves the ID of the image in the same folder in result.txt.

At the same time the console will output the detection time, depending on the resolution of the picture.

