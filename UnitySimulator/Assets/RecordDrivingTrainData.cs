using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Vehicles.Car;
using System.IO;
using System;

public class RecordDrivingTrainData : MonoBehaviour {
    /// <summary>
    /// This script is to record image data of a car by positioning it along a trajectory.
    /// </summary>

    /// for the interface
    public DataRecorder recorder;
    public CarRemoteControl carRemote;
    public string roadType;

    /// for the car
    private List<Vector3> positionList, velocityList;
    private List<float> angleList;

    /// for I/O
    private string trainFolder;
    private int nTrajFrame, rotationAdjust;

    void Start()
    {
        ///*******************************
        /// hardcode values
        ///*******************************
        trainFolder = string.Concat("C:/DeepDrive/train/201809/avoid-", roadType, "/");

        /// initialize variables
        positionList = new List<Vector3>();
        velocityList = new List<Vector3>();
        angleList = new List<float>();

        /// set some variable values
        switch (roadType)
        {
            case "straight":
                nTrajFrame = 73;
                rotationAdjust = -1;
                break;
            case "curved":
                Debug.Log("helloo-curved");
                nTrajFrame = 23;
                rotationAdjust = 1;
                break;
            default:
                throw new Exception("The road type is not specified, enter either straight or curve!");
        }

        /// initialize the recorder
        recorder.Init(carRemote, trainFolder);
        recorder.LoadObstacle(roadType);
        trainFolder = trainFolder + "input/traj/";

        /// record the camera data alone a trajectory
        for (int i = 0; i <= nTrajFrame; i++)
        {
                string trajName = trainFolder + i.ToString() + "-" + 5.ToString();
                LoadTraj (trajName);

                for (int k = 0; k < positionList.Count; k++)
                {
                    float angle = angleList[k];
                    Vector3 pos = positionList[k];
                    float orient = rotationAdjust * Vector3.Angle(Vector3.forward, velocityList[k]);
                    recorder.SetCarPosition(pos);
                    recorder.SetCarOrientation(Quaternion.AngleAxis(orient, transform.up));
                    recorder.WriteTrainData(i, angle, pos, orient);
                }
        }


        /// record the camera data by conducting extra manipulations at each position of a trajectory
        if (false)
        {
            for (int j = 0; j < positionList.Count; j++)
            {
                // horizontal shift
                float startShift = -0.3f;
                float endShift = 0.3f;
                float curShift = startShift;
                float gapShift = 0.1f;
                while (curShift <= endShift)
                {
                    Vector3 pos = positionList[j];
                    pos.x += curShift;
                    recorder.SetCarPosition(pos);

                    // left-to-right rotation
                    float startAngle = -5f;
                    float endAngle = 5f;
                    float curAngle = startAngle;
                    float gapAngle = 1f;
                    while (curAngle <= endAngle)
                    {
                        recorder.SetCarOrientation(Quaternion.AngleAxis(curAngle, Vector3.up));
                        //recorder.WriteTrainData(-1, angleList[j], pos, curAngle);
                        curAngle += gapAngle;
                    }
                    curShift += gapShift;
                }
            }
        }

    }
    

    private void LoadTraj(string traj)
    {
        positionList.Clear();
        velocityList.Clear();
        angleList.Clear();
        StreamReader reader = new StreamReader(traj);
        string line;
        do
        {
            line = reader.ReadLine();
            if (line != null)
            {
                string[] entries = line.Split(',');
                if (entries.Length > 0)
                {
                    positionList.Add(new Vector3(float.Parse(entries[0]), 0f, float.Parse(entries[1])));
                    velocityList.Add(new Vector3(float.Parse(entries[2]), 0f, float.Parse(entries[3])));
                    angleList.Add(float.Parse(entries[4]));
                }
            }
        }
        while (line != null);
        reader.Close();
    }
    
    void Update() { }
}
