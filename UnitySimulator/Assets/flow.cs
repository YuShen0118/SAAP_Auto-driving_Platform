using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class flow : MonoBehaviour {

    // from the right-most lane to the left-most lane
    Object[] carSet;                // car prefabs
    int[] laneCarNum;               // the number of cars on each lane
    float[] laneLeaderPos;          // the position of the ultimate leaders
    float[] laneCtrPos;             // the center position of each lane
    float sedanLength;              // the sedan length
    float[] laneFirstCarPos;         // the position of the first car on each lane

    List<Vector3> lane0Pos;    // current positions (the last is the ultimate leader)
    List<float> lane0Vel;      // current velocities
    List<float> lane0VelGoal;  // goal velocities
    List<float> lane0Len;   // car lengths
    List<GameObject> lane0Obj; // car game objects

    List<Vector3> lane1Pos;    // current positions (the last is the ultimate leader)
    List<float> lane1Vel;      // current velocities
    List<float> lane1VelGoal;  // goal velocities
    List<float> lane1Len;   // car lengths
    List<GameObject> lane1Obj; // car game objects

    List<Vector3> lane2Pos;    // current positions (the last is the ultimate leader)
    List<float> lane2Vel;      // current velocities
    List<float> lane2VelGoal;  // goal velocities
    List<float> lane2Len;   // car lengths
    List<GameObject> lane2Obj; // car game objects

    List<Vector3> lane3Pos;    // current positions (the last is the ultimate leader)
    List<float> lane3Vel;      // current velocities
    List<float> lane3VelGoal;  // goal velocities
    List<float> lane3Len;   // car lengths
    List<GameObject> lane3Obj; // car game objects


    // Use this for initialization
    void Start () {

        //////////////////////////////////
        /// hard-code variables
        //////////////////////////////////
        laneCarNum = new int[] { 0, 1, 0, 0 };
        laneFirstCarPos = new float[] {0f, 0f, 100f, 0f };
        Debug.Log("asdf3");
        //////////////////////////////////

        // init
        initVarables();

        // generate traffic
        for (int i = 0; i < laneCarNum.Length; i++)
        {
            if (laneCarNum[i] > 0)
            {
                switch (i)
                {
                    case 0:
                        generateTraffic(i, ref lane0Pos, ref lane0Vel, ref lane0VelGoal, ref lane0Len, ref lane0Obj);
                        break;
                    case 1:
                        generateTraffic(i, ref lane1Pos, ref lane1Vel, ref lane1VelGoal, ref lane1Len, ref lane1Obj);
                        break;
                    case 2:
                        generateTraffic(i, ref lane2Pos, ref lane2Vel, ref lane2VelGoal, ref lane2Len, ref lane2Obj);
                        break;
                    case 3:
                        generateTraffic(i, ref lane3Pos, ref lane3Vel, ref lane3VelGoal, ref lane3Len, ref lane3Obj);
                        break;
                    default:
                        break;
                }
            }
        }
    }
	
	// Update is called once per frame
	void Update () {

        for (int i = 0; i < laneCarNum.Length; i++)
        {
            if (laneCarNum[i] > 0)
            {
                switch (i)
                {
                    case 0:
                        updateTraffic(i, ref lane0Pos, ref lane0Vel, ref lane0VelGoal, ref lane0Len, ref lane0Obj);
                        break;
                    case 1:
                        updateTraffic(i, ref lane1Pos, ref lane1Vel, ref lane1VelGoal, ref lane1Len, ref lane1Obj);
                        break;
                    case 2:
                        updateTraffic(i, ref lane2Pos, ref lane2Vel, ref lane2VelGoal, ref lane2Len, ref lane2Obj);
                        break;
                    case 3:
                        updateTraffic(i, ref lane3Pos, ref lane3Vel, ref lane3VelGoal, ref lane3Len, ref lane3Obj);
                        break;
                    default:
                        break;
                }
            }
        }
    }

    private void generateTraffic(int laneIdx, ref List<Vector3> lanePos, ref List<float> laneVel, ref List<float> laneVelGoal, ref List<float> laneLength, ref List<GameObject> laneObj)
    {
        GameObject carObj = null;

        int reverseIdx = 1;  // going +z direction
        if (laneIdx > 1)
            reverseIdx = -1; // going -z direction

        for (int i = 0; i < laneCarNum[laneIdx]; i++)
        {
            // sample a car game object
            carObj = (GameObject)Instantiate(carSet[Random.Range(0, carSet.Length)]) as GameObject;
            DestroyImmediate(carObj.GetComponent<Rigidbody>());

            // position the car and rotate the car if necessary 
            if (reverseIdx > 0) // going +z direction
            {
                //carObj.transform.position = new Vector3(laneCtrPos[laneIdx], 0f, -90f);
                carObj.transform.position = new Vector3(laneCtrPos[laneIdx], 0f, -1500f + (i + 1) * 7f + Random.Range(0f, 7f));
            }
            else // going -z direction
            {
                carObj.transform.rotation = Quaternion.AngleAxis(-180, transform.up);
                carObj.transform.position = new Vector3(laneCtrPos[laneIdx], 0f, 1000f - laneFirstCarPos[laneIdx] - (i + 1) * 7f + Random.Range(0f, 7f));
            }

            // name the car
            carObj.name = "agt" + laneIdx.ToString() + "-" + i.ToString();

            // store the specs of this car 
            lanePos.Add(carObj.transform.position);
            laneVel.Add(0f);
            laneVelGoal.Add(Random.Range(15f, 16f));
            //laneVelGoal.Add(10f);
            laneLength.Add(sedanLength);
            laneObj.Add(carObj);
        }

        // create the ultimate leader without a game object
        lanePos.Add(new Vector3(laneCtrPos[laneIdx], 0f, laneLeaderPos[laneIdx]));
        laneVel.Add(0f);
        laneVelGoal.Add(0f);
        laneLength.Add(6f);
    }

    private void updateTraffic(int laneIdx, ref List<Vector3> lanePos, ref List<float> laneVel, ref List<float> laneVelGoal, ref List<float> laneLength, ref List<GameObject> laneObj)
    {
        float tmpAcl = 0;
        float tmpVel = 0;
        float tmpPosZ = 0;

        // going to the opposite direction, we can treat the acceleration and velocity as scalars
        // the only place need to apply any change is when computing tmpPosZ
        int reverseIdx = 1;
        if (laneIdx > 1)
            reverseIdx = -1;

        for (int i = 0; i < laneCarNum[laneIdx]; i++)
        {
            // compute the acceleration according to IDM
            tmpAcl = IDM(laneVel[i], laneVel[i + 1], laneVelGoal[i], lanePos[i].z, lanePos[i + 1].z);

            // update the current velocity
            tmpVel = laneVel[i] + tmpAcl * Time.deltaTime;
            laneVel[i] = tmpVel;

            // update the current position
            tmpPosZ = lanePos[i].z + reverseIdx * tmpVel * Time.deltaTime + 0.5f * reverseIdx * tmpAcl * Time.deltaTime;
            lanePos[i] = new Vector3(laneCtrPos[laneIdx], 0f, tmpPosZ);

            // move the game object 
            laneObj[i].transform.position = lanePos[i];
        }
    }

    private float IDM(float egoVel, float leaderVel, float egoVelGoal, float egoPos, float leaderPos)
    {
        //Debug.Log(egoVel.ToString() + "," + leaderVel.ToString() + "," + egoVelGoal.ToString() + "," + egoPos.ToString() + "," + leaderPos.ToString() + "," + leaderLen.ToString());
        float acl = 0f;

        float sStarSub = egoVel * 1.5f + egoVel * (egoVel - leaderVel) / 2.8f;
        float sStar = 2f + sStarSub > 0? sStarSub:0;

        float sAlpha = Mathf.Abs(leaderPos - egoPos);
        acl = 1f * (1 - Mathf.Pow(egoVel / egoVelGoal, 4) - Mathf.Pow(sStar / sAlpha, 2));
        return acl;
    }

    private void initVarables()
    {
        laneLeaderPos = new float[] { 2000f, 2000f, -2000f, -2000f }; // z values from the right to the left lane
        laneCtrPos = new float[] { 1.875f, -1.875f, -14.625f, -18.375f }; // x values from the right to the left lane
        sedanLength = 7f;
        carSet = Resources.LoadAll("CarPrefabsSmall");

        lane0Pos = new List<Vector3>();
        lane0Vel = new List<float>();
        lane0VelGoal = new List<float>();
        lane0Len = new List<float>();
        lane0Obj = new List<GameObject>();

        lane1Pos = new List<Vector3>();
        lane1Vel = new List<float>();
        lane1VelGoal = new List<float>();
        lane1Len = new List<float>();
        lane1Obj = new List<GameObject>();

        lane2Pos = new List<Vector3>();
        lane2Vel = new List<float>();
        lane2VelGoal = new List<float>();
        lane2Len = new List<float>();
        lane2Obj = new List<GameObject>();

        lane3Pos = new List<Vector3>();
        lane3Vel = new List<float>();
        lane3VelGoal = new List<float>();
        lane3Len = new List<float>();
        lane3Obj = new List<GameObject>();
    }
}
