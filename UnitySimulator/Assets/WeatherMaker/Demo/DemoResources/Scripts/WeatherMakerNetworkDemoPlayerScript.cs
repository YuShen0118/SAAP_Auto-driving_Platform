using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerNetworkDemoPlayerScript : NetworkBehaviour
    {
        [SerializeField]
        private Object[] NetworkObjectsToDelete;

    	private void Start()
        {
            // cleanup networked players of cameras, audio listener, etc.
            if (!isLocalPlayer)
            {
                foreach (Object obj in NetworkObjectsToDelete)
                {
                    Object.Destroy(obj);
                }
                NetworkObjectsToDelete = null;
            }
    	}
    	
    	private void Update()
        {
    		
    	}
    }
}