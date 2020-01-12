using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class curve_wp : MonoBehaviour {


    private List<Vector3> wpPos;
    private Vector3 wpTmp;
    Object[] carSet;
    GameObject carObj;
    private int posIdx;

	// Use this for initialization
	void Start () {

        Debug.Log("asdf3");

        wpPos = new List<Vector3>();
        carSet = Resources.LoadAll("CarPrefabsSmall");
        carObj = (GameObject)Instantiate(carSet[Random.Range(0, carSet.Length)]) as GameObject;
        DestroyImmediate(carObj.GetComponent<Rigidbody>());

        float radius = 51.875f;


        ////////////////////////////////////
        ////    straight road
        float gap1 = 0.8f;
        float curZ = -100f;
        while (curZ <= -1)
        {
            wpPos.Add(new Vector3(-radius, 0f, curZ));
            //GameObject wpTemplate = (GameObject)GameObject.CreatePrimitive(PrimitiveType.Cube);
            //wpTemplate.transform.position = new Vector3(tmpX, 0f, curZ);
            curZ += gap1;
        }


        ////////////////////////////////////
        ////    curved road
        float curAngle = -Mathf.PI;
        float gap2 = 0.02f;
        while (curAngle <= 0)
        {
            wpPos.Add(new Vector3(radius * Mathf.Cos(curAngle), 0f, -radius * Mathf.Sin(curAngle)));
            //GameObject wpTemplate = (GameObject)GameObject.CreatePrimitive(PrimitiveType.Cube);
            //wpTemplate.transform.position = new Vector3(tmpX, 0f, tmpZ);
            curAngle += gap2;
        }

        carObj.transform.position = wpPos[0];
        posIdx = 0;


        ////////////////////////////////////
        ////    straight road on the other side
        curZ = -1f;
        while (curZ >= -100f)
        {
            wpPos.Add(new Vector3(radius, 0f, curZ));
            //GameObject wpTemplate = (GameObject)GameObject.CreatePrimitive(PrimitiveType.Cube);
            //wpTemplate.transform.position = new Vector3(tmpX, 0f, curZ);
            curZ -= gap1;
        }

        //carObj.transform.rotation = Quaternion.AngleAxis(45, Vector3.up);
    }
	
	// Update is called once per frame
	void Update () {


        posIdx++;

        Vector3 v1;
        Vector3 v2;
        float angB = 0;

        if (posIdx > 0 && posIdx < wpPos.Count - 1)
        {

            //v1 = wpPos[posIdx] - wpPos[posIdx - 1];
            v2 = wpPos[posIdx + 1] - wpPos[posIdx];

            //v1.Normalize();
            v2.Normalize();

            angB = Vector3.Angle(Vector3.forward, v2);

            carObj.transform.position = wpPos[posIdx];

            //Debug.Log(angB);
            carObj.transform.rotation = Quaternion.AngleAxis(angB, Vector3.up);


        }


    }
}
