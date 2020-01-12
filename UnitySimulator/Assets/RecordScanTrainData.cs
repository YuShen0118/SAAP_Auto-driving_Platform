using System.Collections;
using System.Collections.Generic;
using UnityStandardAssets.Vehicles.Car;
using UnityEngine;

public class RecordScanTrainData : MonoBehaviour {

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
        trainFolder = string.Concat("C:/DeepDrive/train/ICRA19/detect-", roadType, "/");
        recorder.Init(carRemote, trainFolder);
        recorder.LoadObstacle(roadType);

        float minx = -3f;
        float maxx = 3f;
        float curx = minx;
        float gapx = 0.02f;

        float minz = 50f;
        float maxz = 90f;
        float curz = minz;
        float gapz = 0.5f;


        while(curz <= maxz)
        {
            while(curx <= maxx)
            {
                Vector3 pos = new Vector3(curx, 0f, curz);
                recorder.SetCarPosition(pos);

                if(curx >= -1.2)
                    recorder.WriteTrainData(1);
                else
                    recorder.WriteTrainData(0);

                curx += gapx;
            }

            curx = minx;
            curz += gapz;
        }

    }

    void Update() { }
}
