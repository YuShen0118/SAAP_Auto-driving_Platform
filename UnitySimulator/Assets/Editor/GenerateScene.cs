using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
using GSD.Roads;
using GSD;

public class GenerateScene : MonoBehaviour {
    
    [MenuItem("RoadNetwork/Generate")]
    static void Create()
    {
        //****************************************************************
        // variables
        string netPath = "C:/DeepDrive/data/road_networks/halfCircle/";
        int roadNum = 1;
        bool createNetworkFlag = true;
        //int roadNum = Directory.GetFiles(netPath).Length - 1;
        //****************************************************************


        // Create a terrain that is big enough to hold the road network
        // (requires "RoadBounds" in the above folder)
        StreamReader reader = new StreamReader(netPath + "RoadBounds");
        string line = reader.ReadLine();
        string[] words = line.Split(',');
        Debug.Log("Terrain bounds: minx=" + words[0] + ", maxx=" + words[1] + ", minz=" + words[2] + ", maxz=" + words[3]);
        float minx = float.Parse(words[0]);
        float maxx = float.Parse(words[1]);
        float minz = float.Parse(words[2]);
        float maxz = float.Parse(words[3]);
        double transx = minx * 1.5;
        double transz = minz * 1.5;
        float ctrz = (minz + maxz) / 2;
        double twidth = (maxx - minx) * 1.5;   // make the terrain 1.5x bigger
        double tlength = (maxz - minz) * 1.5;
        int twidthInt = (int)twidth;
        int tlengthInt = (int)tlength;
        CreateTerrain(twidthInt, tlengthInt, transx, transz);
        reader.Close();


        // Create a road network
        if (createNetworkFlag)
        {
            string roadSysStr = "RoadArchitectSystem1";

            // Get the existing road system (if it exists) and destory its consisting roads
            GameObject exRoadSystemObj = (GameObject)GameObject.Find(roadSysStr);
            if (exRoadSystemObj != null)
            {
                Object[] tRoads = exRoadSystemObj.GetComponents<GSDRoad>();
                foreach (GSDRoad xRoad in tRoads)
                {
                    GSD.Roads.GSDTerraforming.TerrainsReset(xRoad);
                }
                Object.DestroyImmediate(exRoadSystemObj);
            }

            // Create a new road system and turn off its updates (only generating the nodes but not the mesh)
            GameObject roadSystemObj = new GameObject(roadSysStr);
            GSDRoadSystem roadSystem = roadSystemObj.AddComponent<GSDRoadSystem>();    //Add road system component.
            roadSystem.opt_bAllowRoadUpdates = false; // this has to be set to false in order for following functions to operate

            // Create a road network 
            CreateRoadNetwork(roadSystem, netPath, roadNum);

            // Turn updates back on and update the roads (create meshes for connecting the road nodes)
            roadSystem.opt_bAllowRoadUpdates = true;
            roadSystem.UpdateAllRoads();
        }

    }

    static void CreateTerrain(int width, int length, double transx, double transz)
    {
        // Destory the previously created terrain
        GameObject exTerrain = (GameObject)GameObject.Find("Terrain");
        if (exTerrain != null)
            DestroyImmediate(exTerrain);


        // Create a new terrain with specs
        GameObject TerrainObj = new GameObject("Terrain");
        TerrainData _TerrainData = new TerrainData();

        _TerrainData.size = new Vector3(width / 16f, 600, length / 16f); // for some reason, the width and length need to divided by 16
        _TerrainData.heightmapResolution = 512;
        _TerrainData.baseMapResolution = 1024;
        _TerrainData.SetDetailResolution(1024, 8);

        TerrainCollider _TerrainCollider = TerrainObj.AddComponent<TerrainCollider>();
        Terrain _Terrain2 = TerrainObj.AddComponent<Terrain>();

        _TerrainCollider.terrainData = _TerrainData;
        _Terrain2.terrainData = _TerrainData;

        TerrainObj.transform.Translate((float)transx, 0, (float)transz);

    }

    static void CreateRoadNetwork(GSDRoadSystem roadSystem, string netPath, int roadNum)
    {
        for (int i=1; i<= roadNum; i++)
        {
            string fileName = netPath + "/Road" + i.ToString();

            StreamReader reader = new StreamReader(fileName);
            string line = reader.ReadLine();

            List<Vector3> nodePos = new List<Vector3>();
            while (line != null)
            {
                string[] words = line.Split(',');
                nodePos.Add(new Vector3(float.Parse(words[0]), float.Parse(words[1]), float.Parse(words[2])));
                line = reader.ReadLine();
            }
            reader.Close();

            GSDRoad tmpRoad = null;
            tmpRoad = GSDRoadAutomation.CreateRoad_Programmatically(roadSystem, ref nodePos);

            // TODO: the generation of intersection doesn't work very well at this point
            // Weizi: seems like an intersection will only get created when two nodes are within 3.3 distance
            //GSDRoadAutomation.CreateIntersections_ProgrammaticallyForRoad(tmpRoad, GSDRoadIntersection.iStopTypeEnum.TrafficLight1, GSDRoadIntersection.RoadTypeEnum.NoTurnLane); // GSDRoadIntersection.iStopTypeEnum.TrafficLight2 doesn't work
        }


        ///////////////////////////
        // Debug file reading
        ///////////////////////////
        /*int hCount = nodePos.Count;
        Debug.Log(hCount.ToString());
        for (int i = 0; i < hCount; i++)
        {
            string ot = nodePos[i].x.ToString() + " " + nodePos[i].y.ToString() + " " + nodePos[i].z.ToString() + "\n";
            Debug.Log(ot);
        }*/


        ///////////////////////////
        // Test road and intersection creation by manually specifying some road nodes
        ///////////////////////////
        /*float StartLocX = 800f;
        float StartLocZ = 200f;
        float GapZ = 200f;
        float tHeight = 0.03f;
        GSDRoad bRoad = null;


        // Create roads
        List<Vector3> nodePos = new List<Vector3>();
        for (int i = 0; i < 5; i++)
        {
            nodePos.Add(new Vector3(StartLocX + (i * 200f), tHeight, 600f));
        }
        bRoad = GSDRoadAutomation.CreateRoad_Programmatically(roadSystem, ref nodePos);

        nodePos.Clear();
        for (int i = 0; i < 5; i++)
        {
            nodePos.Add(new Vector3(StartLocX, tHeight, StartLocZ + (i * GapZ) + 3.3f));
        }
        bRoad = GSDRoadAutomation.CreateRoad_Programmatically(roadSystem, ref nodePos);

        // Create intersections (TrafficLight2 doesn't work)
        // Weizi: seems like an intersection will only get created when two nodes are within 3.3 distance
        GSDRoadAutomation.CreateIntersections_ProgrammaticallyForRoad(bRoad, GSDRoadIntersection.iStopTypeEnum.TrafficLight1, GSDRoadIntersection.RoadTypeEnum.NoTurnLane);*/

    }


}
