using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using System.Text;
using UnityEngine;

public class AllDataCapture : MonoBehaviour
{
    public string outputPath = "D:/projects/gitProjects/SAAP_Auto-driving_Platform/Data/training_simu_1/";

    public Camera mainCam; //待截图的目标摄像机

    public bool captureImage = true;
    public bool captureDepthMap = true;
    public bool capturePointCloud = true;
    public bool copyCameraCalibration = true;
    public bool saveEnd2EndLabel = true;

    public bool speedupCapture = true;

    bool isImageMode = true;

    GameObject mainCamObj;
    RenderTexture rt;  //声明一个截图时候用的中间变量 
    Texture2D t2d;
    float[] pointCloud;
    int imgWidth = 1242;
    int imgHeight = 375;
    int num = 0;  //截图计数

    Shader rgbShader;
    Shader depthShader;
    Renderer rend;

    GameObject[] obj; //开头定义GameObject数组

    // Use this for initialization
    void Start()
    {
        if (mainCam == null)
        {
            mainCamObj = GameObject.Find("CenterCamera");
            init(GetComponent<Renderer>(), mainCamObj.GetComponent<Camera>());
        }
        else
        {
            init(GetComponent<Renderer>(), mainCam);
        }
    }

    public void init(Renderer renderer, Camera camera)
    {
        t2d = new Texture2D(imgWidth, imgHeight, TextureFormat.RGB24, false);
        rt = new RenderTexture(imgWidth, imgHeight, 24);
        mainCam = camera;
        mainCam.targetTexture = rt;


        rend = renderer;
        //rend = GetComponentInChildren<Renderer>();
        rgbShader = Shader.Find("Standard");
        depthShader = Shader.Find("Custom/DepthGrayscale");

        pointCloud = new float[rt.height * rt.width * 4];

        if (saveEnd2EndLabel)
        {
            //clear history
            File.WriteAllText(outputPath + "/end2endLabels.csv", "");
        }
    }

    public byte[] getRenderResult()
    {
        mainCam.targetTexture = rt;
        RenderTexture.active = mainCam.targetTexture;
        mainCam.Render();
        t2d.ReadPixels(new Rect(0, 0, mainCam.targetTexture.width, mainCam.targetTexture.height), 0, 0);
        t2d.Apply();
        return t2d.EncodeToJPG();
    }

    public byte[] getPointCloud(int step = 2)
    {
        int pos = 0;
        for (int i = 0; i < rt.height; i++)
        {
            for (int j = 0; j < rt.width; j++)
            {
                Color c = t2d.GetPixel(j, rt.height - 1 - i);//获取图片xy坐标处的颜色

                if (i % step != 0 || j % step != 0) continue;

                double x, y, z, range;
                // Obtain X and Y Pixel Coordinates
                double pixelX = j;
                double pixelY = i;
                range = c.r * mainCam.farClipPlane;
                x = pixelX - rt.width * 0.5f;
                y = pixelY - rt.height * 0.5f;
                double Fov = mainCam.fieldOfView;
                z = (double)(rt.height / (double)2 / (Math.Tan((Fov / 2) * Math.PI / 180)));

                // unitize the vector
                x /= z;
                y /= z;
                z /= z;
                // multiply vector components by range to obtain real x, y, z

                //int pos = rt.width * i * 4 + j * 4;
                //                     pointCloud[pos] = x * range;
                //                     pointCloud[pos + 1] = y * range;
                //                     pointCloud[pos + 2] = z * range;
                //                     pointCloud[pos + 3] = 1;

                if (-y * range < 3)
                {
                    pointCloud[pos++] = (float)(z * range);
                    pointCloud[pos++] = (float)(-x * range);
                    pointCloud[pos++] = (float)(-y * range);
                    pointCloud[pos++] = 1;
                }
            }
        }

        byte[] pcByteArray = new byte[pos * 4];
        Buffer.BlockCopy(pointCloud, 0, pcByteArray, 0, pcByteArray.Length);

        return pcByteArray;
    }

    public void changeShader(Shader newShader)
    {
        obj = FindObjectsOfType(typeof(GameObject)) as GameObject[]; //get all objects
        foreach (GameObject child in obj)
        {
            //Debug.Log(child.gameObject.name);

            rend = child.GetComponent<Renderer>();
            if (rend == null)
            {
                rend = child.GetComponentInChildren<Renderer>();
            }

            if (rend != null)
            {
                foreach (Material material in rend.materials)
                {
                    material.shader = newShader;
                }
            }
        }
    }
    
    public void WriteTrainData(string fileName, string centerImageName, float label)
    {
        string imgC = centerImageName;
        string imgL = "";
        string imgR = "";
        string rowStr = string.Format("{0},{1},{2},{3}\n", imgC, imgL, imgR, label);
        File.AppendAllText(fileName, rowStr);
    }

    private void Update()
    {
        int startFrame = 10;
        int offset = 0;
        if (num > startFrame)
        {
            int num1 = num - startFrame + offset;
            string imageName = num1.ToString().PadLeft(6, '0') + ".jpg";
            string depthMapName = num1.ToString().PadLeft(6, '0') + ".jpg";
            string pointCloudName = num1.ToString().PadLeft(6, '0') + ".bin";

            if (speedupCapture)
            {
                //only transfer mode for once in one frame, either from rgb to depth or from depth to rgb, to reduce computation
                if (isImageMode)
                {
                    //RGB image capture
                    if (captureImage)
                    {
                        //changeShader(rgbShader);
                        byte[] image = getRenderResult();
                        File.WriteAllBytes(outputPath + "//image_2//" + imageName, image);
                    }

                    if (captureDepthMap || capturePointCloud)
                    {
                        changeShader(depthShader);
                        isImageMode = false;

                        //depth map capture
                        if (captureDepthMap)
                        {
                            byte[] imageDepth = getRenderResult();
                            File.WriteAllBytes(outputPath + "//depth//" + depthMapName, imageDepth);
                        }

                        //point cloud capture
                        if (capturePointCloud)
                        {
                            byte[] pcByteArray = getPointCloud();
                            File.WriteAllBytes(outputPath + "//velodyne//" + pointCloudName, pcByteArray);
                        }
                    }
                }
                else
                {
                    if (captureDepthMap || capturePointCloud)
                    {
                        //depth map capture
                        if (captureDepthMap)
                        {
                            byte[] imageDepth = getRenderResult();
                            File.WriteAllBytes(outputPath + "//depth//" + depthMapName, imageDepth);
                        }

                        //point cloud capture
                        if (capturePointCloud)
                        {
                            byte[] pcByteArray = getPointCloud();
                            File.WriteAllBytes(outputPath + "//velodyne//" + pointCloudName, pcByteArray);
                        }
                    }

                    //RGB image capture
                    if (captureImage)
                    {
                        changeShader(rgbShader);
                        isImageMode = true;
                        byte[] image = getRenderResult();
                        File.WriteAllBytes(outputPath + "//image_2//" + imageName, image);
                    }
                }
            }
            else
            {
                //always change the shader back to rgb.

                //RGB image capture
                if (captureImage)
                {
                    //changeShader(rgbShader);
                    //isImageMode = true;
                    byte[] image = getRenderResult();
                    File.WriteAllBytes(outputPath + "//image_2//" + imageName, image);
                }

                if (captureDepthMap || capturePointCloud)
                {
                    changeShader(depthShader);
                    isImageMode = false;

                    //depth map capture
                    if (captureDepthMap)
                    {
                        byte[] imageDepth = getRenderResult();
                        File.WriteAllBytes(outputPath + "//depth//" + depthMapName, imageDepth);
                    }

                    //point cloud capture
                    if (capturePointCloud)
                    {
                        byte[] pcByteArray = getPointCloud();
                        File.WriteAllBytes(outputPath + "//velodyne//" + pointCloudName, pcByteArray);
                    }
                }

                if (!isImageMode)
                {
                    changeShader(rgbShader);
                    isImageMode = true;
                }
            }

            // Copy calibration data
            if (copyCameraCalibration)
            {
                string sourceFile = outputPath + "//calib//base.txt";
                string destFile = outputPath + "//calib//" + num1.ToString().PadLeft(6, '0') + ".txt";
                System.IO.File.Copy(sourceFile, destFile, true);
            }

            if (saveEnd2EndLabel)
            {
                string fileName = outputPath + "end2endLabels.csv";
                UnityStandardAssets.Vehicles.Car.CarController carController =
                    GameObject.Find("MainCar").GetComponent<UnityStandardAssets.Vehicles.Car.CarController>();
                WriteTrainData(fileName, imageName, carController.CurrentSteerAngle);
            }


            //changeShader(rgbShader);
            RenderTexture.active = null;
            mainCam.targetTexture = null;
        }

        num++;

    }
}
