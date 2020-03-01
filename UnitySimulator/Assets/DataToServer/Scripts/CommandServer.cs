/// <summary>
/// This script is used to send car data in real-time to the server
/// </summary>
/// 
using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using SocketIO;
using UnityStandardAssets.Vehicles.Car;
using System;
using System.Security.AccessControl;
using System.IO;

public class CommandServer : MonoBehaviour
{
    public CarRemoteControl mainCar;
    public Camera mainCamera;
    public bool useIRL = true;
    //public bool saveData = false;
    private SocketIOComponent _socket;
    private CarController _carController;

    private AllDataCapture dataCapturer = new AllDataCapture();

    int imgWidth = 1242;
    int imgHeight = 375;

    Shader rgbShader;
    Shader depthShader;
    float[] pointCloud;

    int num = 0;

    // Use this for initialization
    void Start()
    {
        _socket = GameObject.Find("SocketIO").GetComponent<SocketIOComponent>();
        _socket.On("open", OnOpen);
        _socket.On("steer", OnSteer);
        _socket.On("manual", onManual);
        _carController = mainCar.GetComponent<CarController>();

        dataCapturer.init(GetComponent<Renderer>(), mainCamera);
        
        rgbShader = Shader.Find("Standard");
        depthShader = Shader.Find("Custom/DepthGrayscale");
        
        pointCloud = new float[imgHeight * imgWidth * 4];
    }

    // Update is called once per frame according to FPS
    void Update() { }

    void OnOpen(SocketIOEvent obj)
    {
        Debug.Log("Connection Open");
        EmitTelemetry(obj);
    }

    void OnSteer(SocketIOEvent obj)
    {
        JSONObject jsonObject = obj.data;

        if (useIRL)
        {
            float position_x = float.Parse(jsonObject.GetField("mainCar_position_x").str);
            float position_z = float.Parse(jsonObject.GetField("mainCar_position_y").str);
            mainCar.transform.position = new Vector3(position_x, 0, position_z);

            float velocity_x = float.Parse(jsonObject.GetField("mainCar_velocity_x").str);
            float velocity_z = float.Parse(jsonObject.GetField("mainCar_velocity_y").str);
            mainCar.curVelocity = new Vector3(position_x, 0, position_z);

            float mainCar_direction = float.Parse(jsonObject.GetField("mainCar_direction").str);
            mainCar.transform.eulerAngles = new Vector3(mainCar.transform.eulerAngles.x, mainCar_direction, mainCar.transform.eulerAngles.z);
        }
        else
        {
            mainCar.SteeringAngle = float.Parse(jsonObject.GetField("steering_angle").str);
            mainCar.Acceleration = float.Parse(jsonObject.GetField("throttle").str);
        }

        mainCar.UpdatedFlag = true;

        //CarRemoteControl.AddInput = float.Parse(jsonObject.GetField("add_input").str);
        EmitTelemetry(obj);
    }

    void onManual(SocketIOEvent obj)
    {
        EmitTelemetry(obj);
    }
    
    void EmitTelemetry(SocketIOEvent obj)
    {
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            //print("Sending data to the server...");
            // send only if it's not being manually driven
            if ((Input.GetKey(KeyCode.W)) || (Input.GetKey(KeyCode.S)))
            {
                _socket.Emit("telemetry", new JSONObject());
            }
            else
            {
                num++;
//                 //depth image capture
//                 dataCapturer.changeShader(depthShader);
//                 byte[] imageDepth = dataCapturer.getRenderResult();
//                 if (saveData)
//                     File.WriteAllBytes(dataPath + "//depth//" + num.ToString().PadLeft(6, '0') + ".jpg", imageDepth);
// 
//                 //point cloud capture
//                 byte[] pcByteArray = dataCapturer.getPointCloud();
//                 if (saveData)
//                     File.WriteAllBytes(dataPath + "//velodyne//" + num.ToString().PadLeft(6, '0') + ".bin", pcByteArray);


                //RGB image capture
                //dataCapturer.changeShader(rgbShader);
                byte[] image = dataCapturer.getRenderResult();
//                 if (saveData)
//                     File.WriteAllBytes(dataPath + "//image_2//" + num.ToString().PadLeft(6, '0') + ".jpg", image);

                RenderTexture.active = null;
                mainCamera.targetTexture = null;

                // Collect Data from the Car
                Dictionary<string, string> data = new Dictionary<string, string>();
                data["steering_angle"] = _carController.CurrentSteerAngle.ToString("N4");
                data["throttle"] = _carController.AccelInput.ToString("N4");
                data["speed"] = _carController.CurrentSpeed.ToString("N4");

                data["mainCar_position_x"] = mainCar.transform.position.x.ToString("N4");
                data["mainCar_position_y"] = mainCar.transform.position.z.ToString("N4");

//                 data["mainCar_velocity_x"] = _carController.CurrentVelocity.x.ToString("N4");
//                 data["mainCar_velocity_y"] = _carController.CurrentVelocity.z.ToString("N4");
                data["mainCar_velocity_x"] = mainCar.curVelocity.x.ToString("N4");
                data["mainCar_velocity_y"] = mainCar.curVelocity.z.ToString("N4");
                
                data["mainCar_direction"] = mainCar.transform.eulerAngles.y.ToString("N4");
                //Debug.Log("direction " + mainCar.transform.eulerAngles.y);

                data["image"] = Convert.ToBase64String(image);
                //data["point_cloud"] = Convert.ToBase64String(pcByteArray);
                //data["image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(FrontFacingCamera));
                _socket.Emit("telemetry", new JSONObject(data));
            }
        });

    }
}