using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using System.Text;
using UnityEngine;

public class RGBimgCapture : MonoBehaviour
{
    public Camera mainCam; //待截图的目标摄像机
    RenderTexture rt;  //声明一个截图时候用的中间变量 
    Texture2D t2d;
    int num = 0;  //截图计数
    GameObject mainCar;
    GameObject anotherCar;
    GameObject mainCamObj;
    Vector3 mainCarInitPos;
    Vector3 anotherCarInitPos;
    System.Random rdm;
    //string dataPath = "C:\\Users\\weizili\\Kitti\\object\\training\\";
    string dataPath = "D:\\data\\Kitti\\object\\training_simu_0903\\";

    int imgWidth = 1242;
    int imgHeight = 375;
    float f = 699.7595f;

    // Use this for initialization
    void Start()
    {
        t2d = new Texture2D(imgWidth, imgHeight, TextureFormat.RGB24, false);
        rt = new RenderTexture(imgWidth, imgHeight, 24);
        mainCamObj = GameObject.Find("Camera");
        mainCam = mainCamObj.GetComponent<Camera>();
        mainCam.targetTexture = rt;
        mainCar = GameObject.Find("Interceptor_2A");
        anotherCar = GameObject.Find("Sedan_3A");
        mainCarInitPos = mainCar.transform.position;
        anotherCarInitPos = anotherCar.transform.position;
        rdm = new System.Random(44);
        //Shader sd = Shader.Find("Cars");
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


    // Update is called once per frame
    void Update()
    {
        //按下空格键来截图
        //if (Input.GetKeyDown(KeyCode.Space))
        if (num > 10 && num <= 7490)
        {
            num -= 10;

            //将目标摄像机的图像显示到一个板子上
            //pl.GetComponent<Renderer>().material.mainTexture = rt;

            //截图到t2d中
            RenderTexture.active = rt;
            t2d.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            t2d.Apply();
            RenderTexture.active = null;

            //将图片保存起来
            byte[] byt = t2d.EncodeToPNG();
            File.WriteAllBytes(dataPath + "//image_2//" + num.ToString().PadLeft(6, '0') + ".png", byt);


            Vector3 euler = anotherCar.transform.eulerAngles;
            anotherCar.transform.Rotate(-euler);
            Bounds bbx = GetMaxBounds(anotherCar);
            anotherCar.transform.Rotate(euler);
            Vector3 bbox_2 = new Vector3(bbx.extents.x / anotherCar.transform.localScale.x,
                bbx.extents.y / anotherCar.transform.localScale.y,
                bbx.extents.z / anotherCar.transform.localScale.z);

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
            for (int i = 0; i < 8; i++)
            {
                corners3D[i] = getCameraPoint3D(anotherCar.transform, corners3D[i]);
                corners2D.Add(getImagePoint(corners3D[i]));
            }

            float minu = 1e9f;
            float minv = 1e9f;
            float maxu = -1e9f;
            float maxv = -1e9f;

            for (int i = 0; i < 8; i++)
            {
                if (minu > corners2D[i].x) minu = corners2D[i].x;
                if (minv > corners2D[i].y) minv = corners2D[i].y;
                if (maxu < corners2D[i].x) maxu = corners2D[i].x;
                if (maxv < corners2D[i].y) maxv = corners2D[i].y;
            }
            
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
            //Debug.Log("中心点坐标：" + objcenterInCamera.ToString());


            float deltaY = anotherCar.transform.eulerAngles.y - mainCamObj.transform.eulerAngles.y - 90;
            int rd = (int)(deltaY / 360);
            float remain = deltaY - rd * 360;
            while (remain > 180) remain -= 360;
            while (remain < -180) remain += 360;
            double rotate_y = remain / 180.0 * Math.PI;

            double dry = Math.Atan2(-objcenterInCamera.x, objcenterInCamera.z);


            //generate label
            Matrix4x4 obj2cam = mainCamObj.transform.worldToLocalMatrix * anotherCar.transform.localToWorldMatrix;
            string contents;
            contents = "Car ";
            //contents += " 0 0 0 0 0 0 0 ";

            //Float from 0(non - truncated) to 1(truncated), where truncated refers to the object leaving image boundaries
            contents += truncatedRate.ToString() + " ";

            //Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
            contents += "3 ";


            //Observation angle of object, ranging [-pi..pi]
            contents += (rotate_y + dry).ToString() + " ";

            //left, top, right, bottom 
            contents += minuInImg.ToString() + " " + minvInImg.ToString() + " " + maxuInImg.ToString() + " " + maxvInImg.ToString() + " ";


            Vector2 ctImg = getImagePoint(getCameraPoint3D(anotherCar.transform, new Vector3(0, 0, 0)));
            contents += ctImg.x.ToString() + " " + ctImg.y.ToString() + " ";

            /*
            for (int i=0; i<8; i++)
            {
                contents += corners2D[i].x.ToString() + " " + corners2D[i].y.ToString() + " ";
                Debug.Log("corner 3d：" + corners3D[i].ToString());
            }
            */


            //contents += " 1.7 2.24 5.6 "; // height width length
            //contents += " 1.36 1.792 4.48 "; // height width length * 0.8
            // height width length
            contents += bbx.size.y.ToString() + " " + bbx.size.x.ToString() + " " + bbx.size.z.ToString() + " ";

            contents += obj2cam.m03.ToString() + " " + (-obj2cam.m13 + anotherCar.transform.position.y).ToString() + " " + obj2cam.m23.ToString();


            contents += " " + rotate_y.ToString();

            File.WriteAllText(dataPath + "//label_2//" + num.ToString().PadLeft(6, '0') + ".txt", contents);

            // Copy calibration data
            string sourceFile = dataPath + "//calib//base.txt";
            string destFile = dataPath + "//calib//" + num.ToString().PadLeft(6, '0') + ".txt";
            System.IO.File.Copy(sourceFile, destFile, true);

            // Copy calibration data
            sourceFile = dataPath + "//planes//base.txt";
            destFile = dataPath + "//planes//" + num.ToString().PadLeft(6, '0') + ".txt";
            System.IO.File.Copy(sourceFile, destFile, true);

            Debug.Log("当前截图序号为：" + num.ToString());


            //initialize the car position randomly, take effect at next frame
            
            Vector3 delta;
//             float mdx = -(float)(rdm.Next() % 2000 / 100.0);
//             delta.x = mdx;
//             delta.y = 0;
//             delta.z = 0;
//             mainCar.transform.position = mainCarInitPos + delta;
// 
//             delta.x = (float)(rdm.Next() % 4000 / 100.0 - 35) + mdx;
//             delta.y = 0;
//             delta.z = (float)(rdm.Next() % 400 / 100.0 - 2);
//             anotherCar.transform.position = anotherCarInitPos + delta;
//             anotherCar.transform.Rotate(0, (float)rdm.Next(), 0);
            

            num += 10;
        }
        num++;
    }
}
