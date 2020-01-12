Virtual sensor data capture (KITTI format)
Drag the 'AllDataCapture.cs' in Assets/VirtualSensor to any object you want to bind the script, e.g., the 'centerCamera' object by default. 
Which camera?
Choose the 'Main Cam' under the 'All Data Capture' section in the Inspector page of Unity, e.g., the 'centerCamera' object by default. 
What kinds of data?
You can choose to capture the synchronized image, depth map or point cloud in the view of the selected camera every frame. All the data formats are keeping the same with KITTI dataset.
Image capture needs RGB shader, while depth map and point cloud capture need depth shader, you need to notice that changing shader of every object in every frame is a cost operation, so you'd better choose to capture the data with one shader.
If you choose data formats with two different shaders, e.g., image + depth map or image + point cloud or image + depth map + point cloud, the speedup mode will turn on by default, but it will lead to the flash. You can keep turning on this speedup mode if you just want to run scripts, and you can turn it off if you want to capture the data by driving yourself, but will lead to nearly 2x slower speed.
You can choose to copy the camera calibration file from a basic file, to meet the format of KITTI dataset.
How to set output path?
You can set the output path by modify the parameter 'Output Path' under the 'All Data Capture' section in the Inspector page of Unity. You need to copy and paste the example folder and rename the new one to the name you specified. There are some gasic files and folder architecture in the example folder.


How to use expert?
Drag the 'CarRemoteControl.cs' in CarRemoteControl\Scripts to any object you want to bind the script, e.g., the 'MainCar' object by default. Then you can find the option 'Use Expert'. Currently, we use RVO2 library(http://gamma.cs.unc.edu/RVO2/) as the experts. You may need to rewrite parts of the code now to set up the RVO2 environment, like obstacles. Now part of the obstacles are hard coded in the code.

Car remote control script is used to give controlling instructions to the car controller script (it may get the instructions from the remote controller like user, expert planning algorithm or learning algorithm), while car controller script is used to deal with the input controlling instructions and make the car move.


End-to-end learning module
Train
Save training data with da
The data saving code in this part is implemented by the original orther of xxx, who used a buffer to save the controling data sequence and then save the data to disk asynchronously. I'm doubt about the necessity and correctness because the image is updated when the controling data is saving to the disk, instead of pushing into the buffer. The image and the control data may not be synchronized when the buffer size is larger than 1.
PI control

Test
1. Disable data capture script and expert script, add 'CommandServer' script (in DataToServer/Scripts, already added by default), choose 'Main Car' and 'Main Camera'. 
2. Run drive.py in end2end learning module, and then run this unity project. These two program will be connected by socketIO. Don't forget to setup the parameters in drive.py. See details in the README.txt in end-to-end learning module folder.