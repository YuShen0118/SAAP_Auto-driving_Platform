using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChangeStreetLight : MonoBehaviour {

    public bool dayFlag;
    private int intensity = 5;
    private Light lt;
    void Start()
    {

        foreach (GameObject gameObj in GameObject.FindObjectsOfType<GameObject>())
        {
            // find all street lamps
            if (gameObj.name == "Light")
            {
                // for each street lamp find its light component, adjust its intensity or disable it
                lt = gameObj.GetComponent<Light>();

                if(dayFlag)
                    lt.enabled = !lt.enabled;
                else
                    lt.intensity = intensity;
            }
        }
        
        
    }
}
