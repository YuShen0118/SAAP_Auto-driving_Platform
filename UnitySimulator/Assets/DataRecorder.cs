using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Vehicles.Car;
using System.IO;

public class DataRecorder : MonoBehaviour {

    /// <summary>
    /// Functions for recording the training data.  
    /// </summary>


    private CarController carController;
    private Camera cameraCenter, cameraLeft, cameraRight;
    private string trainFolder;


    void Start() {}
    void Update() {}


    public void Init(CarRemoteControl value, string trainPath)
    {
        // set the car controller and cameras
        carController = value.GetComponent<CarController>();
        cameraCenter = GameObject.Find("CenterCamera").GetComponent<Camera>();
        cameraLeft = GameObject.Find("LeftCamera").GetComponent<Camera>();
        cameraRight = GameObject.Find("RightCamera").GetComponent<Camera>();

        trainFolder = trainPath + "input/";
        Debug.Log(trainFolder);


        if (!Directory.Exists(trainFolder + "imgs/"))
            Directory.CreateDirectory(trainFolder + "imgs/");
        else
        {
            Directory.Delete(trainFolder + "imgs/", true);
            Directory.CreateDirectory(trainFolder + "imgs/");
        }
        
    }


    public void Test()
    {
        Debug.Log("I am in DataRecorder.cs!");
    }


    public void SetCarPosition(Vector3 pos)
    {
        carController.transform.position = pos;
    }


    public void SetCarOrientation(Quaternion orient)
    {
        carController.transform.rotation = orient;
    }


    public void LoadObstacle(string roadType, int obstIdx = 1)
    {
        // load an obstacle
        Object[] obstSet = Resources.LoadAll("Const_prop_sets");
        GameObject obstTmp = (GameObject)Instantiate(obstSet[obstIdx]) as GameObject;
        switch (roadType)
        {
            case "straight":
                obstTmp.transform.position = new Vector3(1.875f, 0f, 100f);
                break;
            case "curved":
                obstTmp.transform.position = new Vector3(-24.06f, 0f, 41.68f);
                break;
            default:
                throw new System.Exception("The road type is not specified, enter either straight or curved!");
        }
        obstTmp.transform.localScale = new Vector3(2f, 10f, 2f);
        //DestroyImmediate(obstTmp.GetComponent<BoxCollider>());
        //DestroyImmediate(obstTmp);
    }

    public void WriteTrainData(int frameIdx, float angle, Vector3 pos, float orient, bool useThreeCamera = false)
    {
        //Debug.Log(frameIdx);
        string imgC = WriteImage("C");
        string imgL = "";
        string imgR = "";
        if (useThreeCamera)
        {
            imgL = WriteImage("L");
            imgR = WriteImage("R");
        }
        string rowStr = string.Format("{0},{1},{2},{3},{4},{5},{6},{7}\n", imgC, imgL, imgR, angle, pos.x, pos.z, orient, frameIdx);
        File.AppendAllText(string.Concat(trainFolder, "log.csv"), rowStr);
    }

    public void WriteTrainData(float label)
    {
        string imgC = WriteImage("C");
        string imgL = "";
        string imgR = "";
        string rowStr = string.Format("{0},{1},{2},{3}\n", imgC, imgL, imgR, label);
        File.AppendAllText(string.Concat(trainFolder, "log.csv"), rowStr);
    }


    private string WriteImage(string cameraStr)
    {
        Camera camera;
        switch (cameraStr)
        {
            case "C":
                camera = cameraCenter;
                break;
            case "L":
                camera = cameraLeft;
                break;
            case "R":
                camera = cameraRight;
                break;
            default:
                camera = cameraCenter;
                break;
        }

        camera.Render();
        RenderTexture targetTexture = camera.targetTexture;
        RenderTexture.active = targetTexture;
        Texture2D screenShot = new Texture2D(targetTexture.width, targetTexture.height, TextureFormat.RGB24, false);
        screenShot.ReadPixels(new Rect(0, 0, targetTexture.width, targetTexture.height), 0, 0);
        screenShot.Apply();
        byte[] image = screenShot.EncodeToJPG();
        DestroyImmediate(screenShot);

        string timestamp = System.DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss_fff");
        string imgName = cameraStr + "_" + timestamp + ".jpg";
        string filePath = trainFolder + "imgs/" + imgName;
        File.WriteAllBytes(filePath, image);
        image = null;
        return imgName;
    }


    public int ProcessLabel(string label)
    {
        int labelValue = -1;
        if (label == "correct")
            labelValue = 0;

        if (label == "avoid")
            labelValue = 1;

        return labelValue;
    }


    public int CountFileInDir(DirectoryInfo d)
    {
        int i = 0;
        FileInfo[] fis = d.GetFiles();
        foreach (FileInfo fi in fis)
        {
            if (fi.Name.Contains("traj"))
                i++;
        }
        return i;
    }
}
