using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using System.Text;
using UnityEngine;

public class AllDataCapture : MonoBehaviour
{
    public string outputPath = "C:/Users/Laura Zheng/Documents/Unity/SAAP_Auto-driving_Platform/Data/training_simu_1/";

    public Camera mainCam; //待截图的目标摄像机

    public bool captureImage = true;
    public bool captureDepthMap = true;
    public bool capturePointCloud = true;
    public bool copyCameraCalibration = true;
    public bool saveEnd2EndLabel = true;
    public bool saveBoundingBoxLabel = true;

    public bool speedupCapture = true;

    bool isImageMode = true;

    GameObject mainCamObj;
    RenderTexture rt;  //声明一个截图时候用的中间变量 
    Texture2D t2d;
    float[] pointCloud;
    int imgWidth = 1242;
    int imgHeight = 375;
    int num = 0;  //截图计数

    // Added by Laura for bounding box labels
    GameObject[] cars;
    GameObject[] trams;
    GameObject[] vans;
    float f = 699.7595f;

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

        if (saveBoundingBoxLabel)
        {
            cars = GameObject.FindGameObjectsWithTag("Car");
            trams = GameObject.FindGameObjectsWithTag("Tram");
            vans = GameObject.FindGameObjectsWithTag("Van");
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

        //if (saveEnd2EndLabel)
        //{
            //clear history
        //    File.WriteAllText(outputPath + "/end2endLabels.csv", "");
        //}
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
        string rowStr = string.Format("{0},{1},{2},{3}" + System.Environment.NewLine, imgC, imgL, imgR, label);
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
                        File.WriteAllBytes(outputPath + "//images//" + imageName, image);
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
                        File.WriteAllBytes(outputPath + "//images//" + imageName, image);
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
                    File.WriteAllBytes(outputPath + "//images//" + imageName, image);
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
                string fileName = outputPath + "//end2endLabels.csv";
                UnityStandardAssets.Vehicles.Car.CarController carController =
                    GameObject.Find("MainCar").GetComponent<UnityStandardAssets.Vehicles.Car.CarController>();
                WriteTrainData(fileName, imageName, carController.CurrentSteerAngle);
            }

            if (saveBoundingBoxLabel)
            {
                String contents = getBoundingBox2D(cars, "Car") + getBoundingBox2D(vans, "Van") + getBoundingBox2D(trams, "Tram");
                string fileName = outputPath + "//labels//" + num1.ToString().PadLeft(6, '0') + ".txt";
                StreamWriter writer = new StreamWriter(fileName);
                writer.WriteLine(contents);
                writer.Close();
            }

            //changeShader(rgbShader);
            RenderTexture.active = null;
            mainCam.targetTexture = null;
        }

        num++;

    }

    Bounds GetMaxBounds(GameObject g)
    {
        var b = new Bounds(g.transform.position, Vector3.zero);
        foreach (Renderer r in g.GetComponentsInChildren<Renderer>())
        {
            b.Encapsulate(r.bounds);
        }
        return b;
    }

    bool isInCamera(GameObject obj)
    {
        Transform cameraTransform = mainCam.transform;
        Transform objTransform = obj.transform;
        Vector3 objcenter = new Vector3(0, 0, 0);
        Vector3 objcenterInCamera = cameraTransform.InverseTransformPoint(objTransform.TransformPoint(objcenter));
        //objcenterInCamera.x = -objcenterInCamera.x;
        objcenterInCamera.y = -objcenterInCamera.y;

        if (objcenterInCamera.z < 0 || objcenterInCamera.z > 80)
            return false;

        int u = (int)(objcenterInCamera.x * f / objcenterInCamera.z + imgWidth / 2);
        int v = (int)(objcenterInCamera.y * f / objcenterInCamera.z + imgHeight / 2);

        if (u < 0 || u >= imgWidth || v < 0 || v >= imgHeight)
        {
            return false;
        }

        //Debug.Log("center position in camera frame: " + objcenterInCamera.x.ToString() + " " + objcenterInCamera.y.ToString() + " " + objcenterInCamera.z.ToString());
        //Debug.Log("center position in image: " + u.ToString() + " " + v.ToString());

        return true;
    }

    Vector3 getCameraPoint3D(Transform objTransform, Vector3 objPoint)
    {
        Vector3 objPtInCamera = mainCam.transform.InverseTransformPoint(objTransform.TransformPoint(objPoint));
        //objPtInCamera.x = -objPtInCamera.x;
        objPtInCamera.y = -objPtInCamera.y;

        return objPtInCamera;
    }

    Vector2 getImagePoint(Vector3 objPtInCamera)
    {
        Vector2 ans;
        ans.x = objPtInCamera.x * f / objPtInCamera.z + imgWidth / 2;
        ans.y = objPtInCamera.y * f / objPtInCamera.z + imgHeight / 2;

        return ans;
    }

    // gets bounding box of specified object list, and label name of that list
    // e.g. cars, "Cars"
    // returns string bounding box label
    String getBoundingBox2D(GameObject[] objList, String labelName)
    {
        GameObject anotherCar;
        String bbox_label = "";
        String contents;

        for (int i = 0; i < objList.Length; i++)
        {
            anotherCar = objList[i];

            // if this object is not in the camera view, go on to the next vehicle in list
            if (isInCamera(anotherCar) == false)
            {
                Debug.Log(anotherCar.name + " is not in the camera view.");
                continue;
            }

            Vector3 euler = anotherCar.transform.eulerAngles;
            anotherCar.transform.Rotate(-euler);
            Bounds bbx = GetMaxBounds(anotherCar);
            anotherCar.transform.Rotate(euler);

            Vector3 bbox_2 = new Vector3(bbx.extents.x / anotherCar.transform.localScale.x,
                bbx.extents.y / anotherCar.transform.localScale.y,
                bbx.extents.z / anotherCar.transform.localScale.z);

            // calculate the 3d bounding box coordinates
            List<Vector3> corners3D = new List<Vector3>();
            corners3D.Add(new Vector3(-bbox_2.x, -bbox_2.y + bbox_2.y, -bbox_2.z));
            corners3D.Add(new Vector3(-bbox_2.x, -bbox_2.y + bbox_2.y, bbox_2.z));
            corners3D.Add(new Vector3(-bbox_2.x, bbox_2.y + bbox_2.y, -bbox_2.z));
            corners3D.Add(new Vector3(-bbox_2.x, bbox_2.y + bbox_2.y, bbox_2.z));
            corners3D.Add(new Vector3(bbox_2.x, -bbox_2.y + bbox_2.y, -bbox_2.z));
            corners3D.Add(new Vector3(bbox_2.x, -bbox_2.y + bbox_2.y, bbox_2.z));
            corners3D.Add(new Vector3(bbox_2.x, bbox_2.y + bbox_2.y, -bbox_2.z));
            corners3D.Add(new Vector3(bbox_2.x, bbox_2.y + bbox_2.y, bbox_2.z));

            List<Vector2> corners2D = new List<Vector2>();
            for (int j = 0; j < 8; j++)
            {
                corners3D[j] = getCameraPoint3D(anotherCar.transform, corners3D[j]);
                corners2D.Add(getImagePoint(corners3D[j]));
            }

            float minu = 1e9f;
            float minv = 1e9f;
            float maxu = -1e9f;
            float maxv = -1e9f;

            for (int j = 0; j < 8; j++)
            {
                if (minu > corners2D[j].x) minu = corners2D[j].x;
                if (minv > corners2D[j].y) minv = corners2D[j].y;
                if (maxu < corners2D[j].x) maxu = corners2D[j].x;
                if (maxv < corners2D[j].y) maxv = corners2D[j].y;
            }

            // 2d bounding box safety check; should not 0 or imgWidth because of earlier check
            float minuInImg = minu;
            if (minuInImg < 0) minuInImg = 0;
            if (minuInImg > imgWidth - 1) minuInImg = imgWidth - 1;

            float minvInImg = minv;
            if (minvInImg < 0) minvInImg = 0;
            if (minvInImg > imgHeight - 1) minvInImg = imgHeight - 1;

            float maxuInImg = maxu;
            if (maxuInImg < 0) maxuInImg = 0;
            if (maxuInImg > imgWidth - 1) maxuInImg = imgWidth - 1;

            float maxvInImg = maxv;
            if (maxvInImg < 0) maxvInImg = 0;
            if (maxvInImg > imgHeight - 1) maxvInImg = imgHeight - 1;

            float truncatedRate = (maxuInImg - minuInImg) * (maxvInImg - minvInImg) / (maxu - minu) / (maxv - minv);
            Vector3 objcenter = new Vector3(0, 0, 0);
            Vector3 objcenterInCamera = getCameraPoint3D(anotherCar.transform, objcenter);

            float deltaY = anotherCar.transform.eulerAngles.y - mainCam.transform.eulerAngles.y - 90;
            int rd = (int)(deltaY / 360);
            float remain = deltaY - rd * 360;
            while (remain > 180) remain -= 360;
            while (remain < -180) remain += 360;
            double rotate_y = remain / 180.0 * Math.PI;

            double dry = Math.Atan2(-objcenterInCamera.x, objcenterInCamera.z);
            Matrix4x4 obj2cam = mainCam.transform.worldToLocalMatrix * anotherCar.transform.localToWorldMatrix;

            // label name, e.g. Car, Tram, Van
            contents = labelName + " ";

            //Float from 0(non - truncated) to 1(truncated), where truncated refers to the object leaving image boundaries
            contents += truncatedRate.ToString() + " ";

            //Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
            contents += "3 ";

            //Observation angle of object, ranging [-pi..pi]
            contents += (rotate_y + dry).ToString() + " ";

            // Bounding box -- left, top, right, bottom 
            contents += minuInImg.ToString() + " " + minvInImg.ToString() + " " + maxuInImg.ToString() + " " + maxvInImg.ToString() + " ";


            //Vector2 ctImg = getImagePoint(getCameraPoint3D(anotherCar.transform, new Vector3(0, 0, 0)));
            //contents += ctImg.x.ToString() + " " + ctImg.y.ToString() + " ";
            

            contents += bbx.size.y.ToString() + " " + bbx.size.x.ToString() + " " + bbx.size.z.ToString() + " ";

            contents += obj2cam.m03.ToString() + " " + (-obj2cam.m13 + anotherCar.transform.position.y).ToString() + " " + obj2cam.m23.ToString();

            contents += " " + rotate_y.ToString();

            bbox_label += contents + System.Environment.NewLine;
        }

        return bbox_label;
    }
}
