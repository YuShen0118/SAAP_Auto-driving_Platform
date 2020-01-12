# SAAP_Auto-driving_Platform

Our platform support following kinds of algorithm architecture:
1. Planning
2. End-to-end learning
3. Perception + planning
4. Perception + learning
We have workded version of all these architecture. You can replace any part to whatever algorithm you want to use.


To use the planning architecture, you need to:
1. Set the right parameters in the Unity simulator.
2. Run the unity simulator.
Related module(s): UnitySimulator. 
Details:
In this architecture, the planning algorithm will get environment information from the Unity simulator directly (or hard coded), so no perception module is needed. Drag the 'CarRemoteControl.cs' in UnitySimulator\Assets\CarRemoteControl\Scripts to any object you want to bind the script, e.g., the 'MainCar' object by default. Then you can find the option 'Use Expert', enable this option. Then you can run the simulator.
Currently, we use RVO2 library(http://gamma.cs.unc.edu/RVO2/) as the experts. You may need to rewrite parts of the code now to set up the RVO2 environment, like obstacles, if you change the default environment. Now part of the obstacles are hard coded in the code.


To use end-to-end learning architecture, you need to:
1. Generate end-to-end learning dataset in the Unity simulator we provided.
2. Train the network with the generated data.
3. Run the trained network and the simulator. After the two program connected, the car in the simulator can drive with the control of the network results.
Related module(s): UnitySimulator, End2EndLearning, Data.
Details:
Train
1. Generate the end-to-end learning dataset. Read the 'Virtual sensor data capture' section and do as it said. You just need to enable 'Capture image' and 'Save End 2 End Label', and disable other data format.
2. Set the parameters in train.py in End2EndLearning folder(remenber to check the 'trainPath' at least), and run train.py. The model will be generated under the folder trainPath + 'trainedModels\models-cnn\' by default.
Test
1. Disable data capture script and expert script (you may have enabled them when collecting training data) in the Unity simulator, add 'CommandServer' script (in UnitySimulator\Assets\DataToServer\Scripts, already added by default), choose 'Main Car' and 'Main Camera'. 
2. Run drive.py in End2EndLearning folder, and then run this unity project. These two program will be connected by socketIO. Don't forget to setup the parameters in drive.py, like model path. If they can not connected with each other, check the url setting in the Socket IO Component, set it to 'ws://127.0.0.1:4567/socket.io/?EIO=4&transport=websocket'


To use the perception + planning architecture, you need to:
1. Generate KITTI format dataset for perception module.
2. Train the perception network with the generated data.
3. Set the right parameters in the Unity simulator.
4. Run the trained networks and the simulator. After the three program connected, the car in the simulator can drive with the control of the network results.
Related module(s): UnitySimulator, Perception, Data. 
Details:


To use perception + learning architecture, you need to:
1. Generate KITTI format dataset for perception module.
2. Train the perception network with the generated data.
3. Generate dataset for the learning module.
4. Train the network in learning module with the generated structured data.
5. Run the trained networks and the simulator. After the three program connected, the car in the simulator can drive with the control of the network results.
Related module(s): UnitySimulator, Perception, IRL, Data.
Details:


Virtual sensor data capture (including KITTI object detection format, and other format needed)
Drag the 'AllDataCapture.cs' in Assets/VirtualSensor to any object you want to bind the script, e.g., the 'centerCamera' object by default. Then run the simulator, it will capture the selected data format automatically. You can use the planning algorithm as expert 
Which camera?
Choose the 'Main Cam' under the 'All Data Capture' section in the Inspector page of Unity, e.g., the 'centerCamera' object by default. 
What kinds of data?
You can choose to capture the synchronized image, depth map or point cloud in the view of the selected camera every frame. All of them are keeping the same with KITTI dataset (object detection task).
Image capture needs RGB shader, while depth map and point cloud capture need depth shader, you need to notice that changing shader of every object in every frame is a cost operation, so you'd better choose to capture the data with one shader if it's acceptable.
If you choose data formats with two different shaders, e.g., image + depth map or image + point cloud or image + depth map + point cloud, the speedup mode will turn on by default, but it will lead to the flash. You can keep turning on this speedup mode if you just want to run scripts, and you can turn it off if you want to capture the data by driving yourself, but will lead to nearly 2x slower speed.
You can choose to copy the camera calibration file from a basic file, to meet the format of KITTI dataset.
How to set output path?
You can set the output path by modify the parameter 'Output Path' under the 'All Data Capture' section in the Inspector page of Unity. You need to copy and paste the sample folder (Data\sample) and rename it. There are some basic files and folder architecture in the sample folder.