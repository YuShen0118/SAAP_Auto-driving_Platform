using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;

public class GenerateWaypointSystem : MonoBehaviour
{

    [MenuItem("WaypointSystem/Generate")]
    static void Create()
    {
        // define the list to store all waypoints positions
        List<Vector3> wpPos = new List<Vector3>();

        // delete previous created gameobjects
        GameObject exSystem = (GameObject)GameObject.Find("WaypointSystem");
        DestroyImmediate(exSystem);

        // create the primitive gameobject
        GameObject wpParent = new GameObject("WaypointSystem");
        GameObject wpTemplate = (GameObject)GameObject.CreatePrimitive(PrimitiveType.Cube);
        wpTemplate.gameObject.name = "WaypointTemplate";
        DestroyImmediate(wpTemplate.GetComponent<BoxCollider>());



        /// load external waypoint setup
        /// 
        if (true)
        {
            wpPos.Add(new Vector3(-42f, 0f, -100f));
            wpPos.Add(new Vector3(-47.5f, 0f, -15.48328f));

            string trajFileName = "C:/DeepDrive/data/road_networks/magnet/WaypointSystem";
            Debug.Log(trajFileName);
            StreamReader reader = new StreamReader(trajFileName);
            string line;
            do
            {
                line = reader.ReadLine();
                if (line != null)
                {
                    string[] entries = line.Split(',');
                    if (entries.Length > 0)
                    {
                        wpPos.Add(new Vector3(float.Parse(entries[0]), 0f, float.Parse(entries[2])));
                    }
                }
            }
            while (line != null);
            reader.Close();

            wpPos.Add(new Vector3(47.5f, 0f, -24.06812f));
            wpPos.Add(new Vector3(42f, 0f, -100f));
        }


        /// create a circular road
        /// 
        if (false)
        {
            int numberOfObjects = 120;
            float radius = 501.75f;
            for (int i = 1; i < numberOfObjects + 1; i++)
            {
                float angle = i * Mathf.PI * 2 / numberOfObjects;
                Vector3 pos = new Vector3(Mathf.Cos(angle) * radius, 0.5f, Mathf.Sin(angle) * radius);
                wpPos.Add(pos);
            }
        }
        
        

        /// load external waypoint setup
        /// 
        if (false)
        {
            string trajFileName = "C:\\DeepDrive\\training\\2017_11\\correctionF_1113\\b060_wp_traj";
            Debug.Log(trajFileName);
            StreamReader reader = new StreamReader(trajFileName);
            string line;
            do
            {
                line = reader.ReadLine();
                if (line != null)
                {
                    string[] entries = line.Split(',');
                    if (entries.Length > 0)
                    {
                        wpPos.Add(new Vector3(float.Parse(entries[0]) - 0.1f, 0f, float.Parse(entries[1])));
                    }
                }
            }
            while (line != null);
            reader.Close();
        }


        /// actually create the waypoint system providing wpPos is not empty
        /// 
        for (int i = 0; i < wpPos.Count; i++)
        {
            GameObject ob = (GameObject)Instantiate(wpTemplate);
            if (i < 10)
                ob.gameObject.name = "Waypoint00" + i.ToString();
            else if (i < 100)
                ob.gameObject.name = "Waypoint0" + i.ToString();
            else
                ob.gameObject.name = "Waypoint" + i.ToString();
            ob.transform.parent = wpParent.transform;
            ob.transform.Translate(wpPos[i]);

            // make waypoint object invisible
            DestroyImmediate(ob.GetComponent<MeshRenderer>());
        }


        DestroyImmediate(wpTemplate);
        

    }


}