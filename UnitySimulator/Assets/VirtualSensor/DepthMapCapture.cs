using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using System.Text;
using UnityEngine;

public class DepthMapCapture : MonoBehaviour
{
    public Camera mainCam; //待截图的目标摄像机
    RenderTexture rt;  //声明一个截图时候用的中间变量 
    Texture2D t2d;
    int num = 0;  //截图计数
    GameObject mainCar;
    GameObject anotherCar;
    Vector3 mainCarInitPos;
    Vector3 anotherCarInitPos;
    System.Random rdm;
    //string dataPath = "C:\\Users\\weizili\\Kitti\\object\\training_simu_gene\\";
    string dataPath = "C:\\Users\\weizili\\Kitti\\object\\training_simunew1\\";

    // Use this for initialization
    void Start()
    {
        t2d = new Texture2D(1242, 375, TextureFormat.RGB24, false);
        rt = new RenderTexture(1242, 375, 24);
        GameObject gameObject = GameObject.Find("Camera");
        mainCam = gameObject.GetComponent<Camera>();
        mainCam.targetTexture = rt;
        mainCar = GameObject.Find("Interceptor_2A");
        anotherCar = GameObject.Find("Sedan_3A");
        mainCarInitPos = mainCar.transform.position;
        anotherCarInitPos = anotherCar.transform.position;
        rdm = new System.Random(44);
    }

    // Update is called once per frame
    void Update()
    {
        //按下空格键来截图
        //if (Input.GetKeyDown(KeyCode.Space))
        if (num > 10 && num <= 7490)
        {
            num -= 10;
            //initialize the car position randomly
            Vector3 delta;
            float mdx = -(float)(rdm.Next() % 2000 / 100.0);
            delta.x = mdx;
            delta.y = 0;
            delta.z = 0;
            mainCar.transform.position = mainCarInitPos + delta;

            delta.x = (float)(rdm.Next() % 4000 / 100.0 - 35) + mdx;
            delta.y = 0;
            delta.z = (float)(rdm.Next() % 400 / 100.0 - 2);
            anotherCar.transform.position = anotherCarInitPos + delta;
            anotherCar.transform.Rotate(0, (float)rdm.Next(), 0);


            //将目标摄像机的图像显示到一个板子上
            //pl.GetComponent<Renderer>().material.mainTexture = rt;

            //截图到t2d中
            RenderTexture.active = rt;
            t2d.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            t2d.Apply();
            RenderTexture.active = null;

            //将图片保存起来
            byte[] byt = t2d.EncodeToPNG();
            File.WriteAllBytes(dataPath + "//depth//" + num.ToString().PadLeft(6, '0') + ".png", byt);


            float[] depthArray = new float[rt.height * rt.width];

            float[] pointCloud = new float[rt.height * rt.width * 4];
            int pos = 0;
            for (int i = 0; i < rt.height; i++)
            {
                for (int j = 0; j < rt.width; j++)
                {
                    Color c = t2d.GetPixel(j, rt.height - 1 - i);//获取图片xy坐标处的颜色

                    if (Math.Abs(c.r - c.b) * 255 > 1 || Math.Abs(c.g - c.b) * 255 > 1 || Math.Abs(c.r - c.g) * 255 > 1)
                    {
                        depthArray[rt.width * i + j] = 1;
                        continue;
                    }

                    depthArray[rt.width * i + j] = c.r;

                    if (i % 2 != 0 || j % 2 != 0) continue;

                    float r, g, b;
                    float x, y, z, range;
                    // Obtain X and Y Pixel Coordinates
                    float pixelX = j;
                    float pixelY = i;
                    range = c.r * mainCam.farClipPlane;
                    x = pixelX - rt.width * 0.5f;
                    y = pixelY - rt.height * 0.5f;
                    double Fov = mainCam.fieldOfView;
                    z = (float)(rt.width * 0.25f * (9 / 16.0) / (Math.Tan((Fov / 2) * Math.PI / 180)));

                    //double vecLength = Math.Sqrt((x * x) + (y * y) + (z * z));
                    // r = g = b because we are getting the value from the depth grayscale image
                    r = (int)(c.r * 255);
                    g = (int)(c.g * 255);
                    b = (int)(c.b * 255);
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
                        pointCloud[pos++] = z * range;
                        pointCloud[pos++] = -x * range;
                        pointCloud[pos++] = -y * range;
                        pointCloud[pos++] = 1;
                    }
                }
            }

            byte[] byteArray = new byte[depthArray.Length * 4];
            Buffer.BlockCopy(depthArray, 0, byteArray, 0, byteArray.Length);
            File.WriteAllBytes(dataPath + "//depth//" + num.ToString().PadLeft(6, '0') + ".byt", byteArray);

            byte[] pcByteArray = new byte[pos * 4];
            Buffer.BlockCopy(pointCloud, 0, pcByteArray, 0, pcByteArray.Length);
            File.WriteAllBytes(dataPath + "//velodyne//" + num.ToString().PadLeft(6,'0') + ".bin", pcByteArray);


            Debug.Log("当前截图序号为：" + num.ToString());

            num += 10;
        }
        num++;
    }
}
