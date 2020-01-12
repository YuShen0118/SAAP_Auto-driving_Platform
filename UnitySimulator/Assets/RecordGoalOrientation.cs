using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Vehicles.Car;
using System.IO;

public class RecordGoalOrientation : MonoBehaviour {

    public DataRecorder recorder;
    public CarRemoteControl carRemote;

    private int nSample;
    private Vector3 pos, goalPos, carOrient, goalOrient;
    private float angle;
    private string trainFolder;

    private float maxRange;
    private float maxAngle;
    private float label;
    private float side;

    void Start () {

        goalPos = new Vector3(0f, 0f, 300f);
        nSample = 30000;
        maxRange = 100f;
        maxAngle = 15f;

        for (int i = 0; i<nSample; i++)
        {
             Debug.Log("hello");
            // init
            
            trainFolder = "C:/DeepDrive/train/201806/open/";
            recorder.Init(carRemote, trainFolder);

            // sample a position
            pos = new Vector3(Random.Range(-maxRange, maxRange), 0f, Random.Range(-maxRange, maxRange));
            //pos = new Vector3(0f, 0f, 0f);
            recorder.SetCarPosition(pos);

            // sample a angle
            angle = Random.Range(-maxAngle, maxAngle);
            //angle = -60;
            recorder.SetCarOrientation(Quaternion.AngleAxis(angle, transform.up));

            carOrient = new Vector3(Mathf.Sin(angle * Mathf.Deg2Rad), 0, Mathf.Cos(angle * Mathf.Deg2Rad));
            goalOrient = goalPos - pos;
            side = Mathf.Sign(Vector3.Cross(carOrient, goalOrient).y);
            label = Vector3.Angle(carOrient, goalOrient);


            // write the data
            recorder.WriteTrainData(label*side);


            
        }


    }
	

}
