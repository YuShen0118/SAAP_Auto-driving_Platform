using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Vehicles.Car;
using System.IO;

public class RecordDetectionTrainData : MonoBehaviour {
    /// <summary>
    /// This script is to record image data of a car by positioning in differet zones.
    /// </summary>


    /// for the interface
    public DataRecorder recorder;
    public CarRemoteControl carRemote;
    public string roadType;

    /// for the scenario
    private List<Vector3> positionList, intentDirectionList, limitAngleList;
    private List<float> multList, leftLabelList, rightLabelList;
    private string trainFolder, trainFile;

    /// for detection data
    float limitAngleModified;
    float angleToIntentDirection;
    float rightMostAngle;
    float leftMostAngle;
    float curAngle;
    float gapRot;
    private float eps;
    private float maxRot;


    void Start()
    {
        ///*******************************
        /// hardcode values
        ///*******************************
        trainFolder = string.Concat("C:/DeepDrive/train/201809/detect-", roadType, "/");
        trainFile = "input/area";

        /// initialize variables
        positionList = new List<Vector3>();
        intentDirectionList = new List<Vector3>();
        limitAngleList = new List<Vector3>();
        multList = new List<float>();
        leftLabelList = new List<float>();
        rightLabelList = new List<float>();

        /// initialize the recorder   
        recorder.Init(carRemote, trainFolder);
        recorder.LoadObstacle(roadType);

        /// load the zone information     
        LoadZoneInfo();

        /// main program
        eps = 1;
        maxRot = 2.5f;
        gapRot = 0.1f;
        for (int i = 0; i < positionList.Count; i++)
        {
            /// set the car position
            recorder.SetCarPosition(positionList[i]);

            /// compute the relevant angles
            limitAngleModified = Vector3.Angle(Vector3.right, limitAngleList[i]) + multList[i] * eps;
            angleToIntentDirection = Vector3.Angle(Vector3.right, intentDirectionList[i]);
            rightMostAngle = angleToIntentDirection - maxRot;
            leftMostAngle = angleToIntentDirection + maxRot;
            curAngle = rightMostAngle;

            /// generate training data
            while (curAngle <= leftMostAngle)
            {
                float angle = 90 - curAngle;
                recorder.SetCarOrientation(Quaternion.AngleAxis(angle, Vector3.up));

                // recorder.WriteTrainData(-1, leftLabelList[i], positionList[i], angle);
                if (curAngle > limitAngleModified)
                    recorder.WriteTrainData(leftLabelList[i]);
                else
                    recorder.WriteTrainData(rightLabelList[i]);
                

                curAngle += gapRot;
            }
        }
    }


    private void LoadZoneInfo()
    {
        StreamReader reader = new StreamReader(string.Concat(trainFolder, trainFile));
        string line;
        do
        {
            line = reader.ReadLine();
            if (line != null)
            {
                string[] entries = line.Split(' ');
                if (entries.Length > 0)
                {
                    positionList.Add(new Vector3(float.Parse(entries[0]), 0f, float.Parse(entries[1])));
                    intentDirectionList.Add(new Vector3(float.Parse(entries[2]), 0f, float.Parse(entries[3])));
                    limitAngleList.Add(new Vector3(float.Parse(entries[4]), 0f, float.Parse(entries[5])));
                    multList.Add(float.Parse(entries[6]));

                    int leftLbl = recorder.ProcessLabel(entries[7]);
                    if (leftLbl < 0)
                        Debug.Log("Error of left label, check input-zone.csv!");
                    else
                        leftLabelList.Add(leftLbl);

                    int rightLbl = recorder.ProcessLabel(entries[8]);
                    if (rightLbl < 0)
                        Debug.Log("Error of right label, check input-zone.csv!");
                    else
                        rightLabelList.Add(rightLbl);
                }
            }
        }
        while (line != null);
        reader.Close();
    }

    

    void Update() { }
}
