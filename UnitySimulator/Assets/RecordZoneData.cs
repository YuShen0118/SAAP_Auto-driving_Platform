using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Vehicles.Car;
using System.IO;

public class RecordZoneData : MonoBehaviour {

    /// <summary>
    /// This script is to record image data of a car by positioning in differet zones.
    /// </summary>


   


    /// for the car
    public CarRemoteControl carRemoteCtrl;
    private CarController carCtrl;
    private Camera centerCam;
    private Camera rightCam;
    private Camera leftCam;
    private List<Vector3> carPos;
    private List<Vector3> carIntentDir;
    private List<Vector3> carActualDir;
    private List<float> carMult;
    private List<int> carLeftLbl;
    private List<int> carRightLbl;
    private float eps;


    /// for obstacles
    private Object[] obstAll;
    private GameObject obstTmp;
    private Vector3 obstPos;
    private Vector3 obstScl;
    private int obstIdx;


    /// for inputs and outputs
    string zoneLog;
    string trainFolder;
    string trainImgFolder;
    string trainLog;
    string analysisLog;


    void Start () {

        /// for the car
        carCtrl = carRemoteCtrl.GetComponent<CarController>();
        centerCam = GameObject.Find("CenterCamera").GetComponent<Camera>();
        rightCam = GameObject.Find("RightCamera").GetComponent<Camera>();
        leftCam = GameObject.Find("LeftCamera").GetComponent<Camera>();
        carPos = new List<Vector3>();
        carIntentDir = new List<Vector3>();
        carActualDir = new List<Vector3>();
        carMult = new List<float>();
        carLeftLbl = new List<int>();
        carRightLbl = new List<int>();
        eps = 1;


        /// for obstacles
        /// 
        obstAll = Resources.LoadAll("Const_prop_sets");
        obstIdx = 1;

        obstPos = new Vector3(1.875f, 0f, 100f); // for the straight road: 1.875f,0f,100f
        obstScl = new Vector3(2f, 10f, 2f);

        obstTmp = (GameObject)Instantiate(obstAll[obstIdx]) as GameObject;
        obstTmp.transform.position = obstPos;
        obstTmp.transform.localScale = obstScl;


        /// specify training file directories 
        /// 
        /// Input
        trainFolder = "C:/DeepDrive/train/201805/detection-straight/";
        zoneLog = "input-zone";

        /// Output
        trainImgFolder = "train-imgs/";
        trainLog = "train-log.csv";
        analysisLog = "analysis-log.csv";
        if (!Directory.Exists(trainFolder + trainImgFolder))
            Directory.CreateDirectory(trainFolder + trainImgFolder);


        /// main program
        LoadZoneInfo();
        Debug.Log(carPos.Count);
        for (int i = 0; i < carPos.Count; i++)
        {
            carCtrl.transform.position = carPos[i];
            float limitAngle = Vector3.Angle(Vector3.right, carActualDir[i]) + carMult[i] * eps;

            float angleToIntentialDir = Vector3.Angle(Vector3.right, carIntentDir[i]);
            Debug.Log(angleToIntentialDir);

            float rightMostAngle = angleToIntentialDir - 2.5f;
            float leftMostAngle = angleToIntentialDir + 2.5f;
            float curAngle = rightMostAngle;
            float gapRot = 0.1f;

            while (curAngle <= leftMostAngle)
            {
                carCtrl.transform.rotation = Quaternion.AngleAxis(90 - curAngle, Vector3.up);

                if (curAngle > limitAngle)
                    writeTrainingData(carLeftLbl[i], carPos[i].z);
                else
                    writeTrainingData(carRightLbl[i], carPos[i].z);

                curAngle += gapRot;
            }
        }
    }


    private void LoadZoneInfo()
    {
        StreamReader reader = new StreamReader(string.Concat(trainFolder, "input-zone"));
        string line;
        do
        {
            line = reader.ReadLine();
            if (line != null)
            {
                string[] entries = line.Split(' ');
                if (entries.Length > 0)
                {
                    carPos.Add(new Vector3(float.Parse(entries[0]), 0f, float.Parse(entries[1])));
                    carIntentDir.Add(new Vector3(float.Parse(entries[2]), 0f, float.Parse(entries[3])));
                    carActualDir.Add(new Vector3(float.Parse(entries[4]), 0f, float.Parse(entries[5])));
                    carMult.Add(float.Parse(entries[6]));

                    int leftLbl = processLabel(entries[7]);
                    if (leftLbl < 0)
                        Debug.Log("Error of left label, check input-zone.csv!");
                    else
                        carLeftLbl.Add(leftLbl);

                    int rightLbl = processLabel(entries[8]);
                    if (rightLbl < 0)
                        Debug.Log("Error of right label, check input-zone.csv!");
                    else
                        carRightLbl.Add(rightLbl);
                }
            }
        }
        while (line != null);
        reader.Close();
    }

    private int processLabel(string lbl)
    {
        int lblValue = -1;
        if (lbl == "correct")
            lblValue = 0;

        if (lbl == "avoid")
            lblValue = 1;

        return lblValue;
    }

    private void writeTrainingData(int frameLabel, float carPosZ)
    {

        //if(frameLabel == 1)
        //{
        //    if (carPosZ > 65 && carPosZ <= 80)
        //        frameLabel = 2;
        //    else if (carPosZ > 80)
        //        frameLabel = 3;
        //}

        string centerImgPath = WriteImage(centerCam, "center", System.DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss_fff"));
        //string leftPath = WriteImage(leftCam, "left", timeStamp);
        //string rightPath = WriteImage(rightCam, "right", timeStamp);
        string leftImgPath = "";
        string rightImgPath = "";
        string row = string.Format("{0},{1},{2},{3}\n", centerImgPath, leftImgPath, rightImgPath, frameLabel);
        File.AppendAllText(string.Concat(trainFolder, trainLog), row);
    }

    private void writeAnalysisData(float frameLabel, Vector3 carPosSgl, float carAngle)
    {
        string row = string.Format("{0},{1},{2},{3},{4}\n", frameLabel, carPosSgl.x, carPosSgl.y, carPosSgl.z, carAngle);
        File.AppendAllText(string.Concat(trainFolder, analysisLog), row);
    }


    private string WriteImage(Camera camera, string prepend, string timestamp)
    {
        camera.Render();
        RenderTexture targetTexture = camera.targetTexture;
        RenderTexture.active = targetTexture;
        Texture2D screenShot = new Texture2D(targetTexture.width, targetTexture.height, TextureFormat.RGB24, false);
        screenShot.ReadPixels(new Rect(0, 0, targetTexture.width, targetTexture.height), 0, 0);
        screenShot.Apply();
        byte[] image = screenShot.EncodeToJPG();
        DestroyImmediate(screenShot);

        string filePath = trainFolder + trainImgFolder + prepend + "_" + timestamp + ".jpg";
        File.WriteAllBytes(filePath, image);
        image = null;
        return filePath;
    }


    void Update () {}
}
