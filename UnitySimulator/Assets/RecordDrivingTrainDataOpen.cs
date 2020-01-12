using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Vehicles.Car;
using System.IO;
using System;

public class RecordDrivingTrainDataOpen : MonoBehaviour {

    /// <summary>
    /// This script is to record image data of a car in the open scenario where the obstacle is a moving car.
    /// </summary>

    /// for the interface
    public DataRecorder recorder;
    public CarRemoteControl carRemote;
    private GameObject obstCar;

    /// for the car
    private List<Vector3> positionList, velocityList;
    private List<float> angleList;

    private List<Vector3> positionListAgent, velocityListAgent;
    private List<float> angleListAgent;

    /// for I/O
    private string trainPath;
    private int nTrajFrame, rotationAdjust;

    void Start()
    {
        obstCar = GameObject.FindGameObjectWithTag("obstCar");
        Debug.Log("hellooo");


        ///*******************************
        /// hardcode values
        ///*******************************
        trainPath = string.Concat("C:/DeepDrive/train/201809/test-scene/");
        nTrajFrame = 0;
        rotationAdjust = -1;
        ///*******************************
        ///*******************************


        /// initialize variables
        positionList = new List<Vector3>();
        velocityList = new List<Vector3>();
        angleList = new List<float>();

        positionListAgent = new List<Vector3>();
        velocityListAgent = new List<Vector3>();
        angleListAgent = new List<float>();

        /// initialize the recorder
        recorder.Init(carRemote, trainPath);

        /// record the camera data alone a trajectory
        for (int i = 0; i <= nTrajFrame; i++)
        {
            string agtTrajFile = trainPath + "input/traj-agt/" + i.ToString() + "-" + 1.ToString();
            LoadTraj(agtTrajFile, ref positionListAgent, ref velocityListAgent, ref angleListAgent);

            string egoTrajFile = trainPath + "input/traj-ego/" + i.ToString() + "-" + 5.ToString();
            LoadTraj(egoTrajFile, ref positionList, ref velocityList, ref angleList);

            for (int k = 0; k < positionList.Count; k++)
            {
                // make sure the tag "obstCar" is applied to the obstacle car model in Unity.
                obstCar.transform.position = positionListAgent[k];
                obstCar.transform.rotation = Quaternion.AngleAxis(-90, transform.up);

                float angle = angleList[k];
                Vector3 pos = positionList[k];
                float orient = rotationAdjust * Vector3.Angle(Vector3.forward, velocityList[k]);
                recorder.SetCarPosition(pos);
                recorder.SetCarOrientation(Quaternion.AngleAxis(orient, transform.up));
                recorder.WriteTrainData(i, angle, pos, orient);
            }
        }
    }


    private void LoadTraj(string trajPath, ref List<Vector3> posList, ref List<Vector3> velList, ref List<float> angList)
    {
        posList.Clear();
        velList.Clear();
        angList.Clear();
        StreamReader reader = new StreamReader(trajPath);
        string line;
        do
        {
            line = reader.ReadLine();
            if (line != null)
            {
                string[] entries = line.Split(',');
                if (entries.Length > 0)
                {
                    posList.Add(new Vector3(float.Parse(entries[0]), 0f, float.Parse(entries[1])));
                    velList.Add(new Vector3(float.Parse(entries[2]), 0f, float.Parse(entries[3])));
                    angList.Add(float.Parse(entries[4]));
                }
            }
        }
        while (line != null);
        reader.Close();
    }

    void Update() { }
}
