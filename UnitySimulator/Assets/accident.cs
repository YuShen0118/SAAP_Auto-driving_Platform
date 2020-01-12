using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class accident : MonoBehaviour {

    private GameObject[] agents;
    private float maxSpeed, accel;
    private List<Vector3> prePos;
    private string rowStr;
    private int frameIdx;
    private Vector3 position;
    private Vector3 velocity;
    private bool writeTraj;
    
    void Start () {

        writeTraj = true;
        maxSpeed = 20f;
        accel = 5;
        prePos = new List<Vector3>();
        rowStr = "";
        frameIdx = 0;
        velocity = new Vector3();

        if (agents == null)
            agents = GameObject.FindGameObjectsWithTag("Player");

        Debug.Log("hello");
        Debug.Log(agents.Length);
    }

    void FixedUpdate() {

        if(!writeTraj)
        {
            for (int i = 0; i < agents.Length; i++)
                agents[i].transform.Translate(Vector3.forward * maxSpeed * Time.deltaTime);
        }
        else
        {
            frameIdx++;
            if (frameIdx > 1)
            {
                for (int i = 0; i < agents.Length; i++)
                {
                    agents[i].transform.Translate(Vector3.forward * maxSpeed * Time.deltaTime);
                    position = agents[i].transform.position;
                    velocity = (position - prePos[i]) / Time.deltaTime;
                    rowStr = string.Format("{0},{1},{2},{3},{4},{5},{6}\n", i, frameIdx - 1, position.x, position.z, velocity.x, velocity.z, System.DateTime.Now.ToString("ss.ffffff"));
                    //File.AppendAllText("C:/DeepDrive/accident-traj.csv", rowStr);
                    prePos[i] = position;
                }
            }
            else
            {
                for (int i = 0; i < agents.Length; i++)
                    prePos.Add(agents[i].transform.position);
            }
        }

       


        

        //foreach (GameObject agent in agents)
        //{
        //    //Rigidbody ab = agent.GetComponent<Rigidbody>();
        //    //if (ab.velocity.magnitude <= maxSpeed)
        //    //    ab.AddRelativeForce(Vector3.forward * accel);
        //    //agent.transform.position += new Vector3(0.0f, 0.0f, maxSpeed * Time.deltaTime);
        //    
        //   Debug.Log(agent.transform.position.x);
        //}
        
        
    }
}
