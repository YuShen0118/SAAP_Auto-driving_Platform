//
// Weather Maker for Unity
// (c) 2016 Digital Ruby, LLC
// Source code may be used for personal or commercial projects.
// Source code may NOT be redistributed or sold.
// 
// *** A NOTE ABOUT PIRACY ***
// 
// If you got this asset off of leak forums or any other horrible evil pirate site, please consider buying it from the Unity asset store at https ://www.assetstore.unity3d.com/en/#!/content/60955?aid=1011lGnL. This asset is only legally available from the Unity Asset Store.
// 
// I'm a single indie dev supporting my family by spending hundreds and thousands of hours on this and other assets. It's very offensive, rude and just plain evil to steal when I (and many others) put so much hard work into the software.
// 
// Thank you.
//
// *** END NOTE ABOUT PIRACY ***
//

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    [RequireComponent(typeof(BoxCollider))]
    public class WeatherMakerNullFogZoneScript : MonoBehaviour
    {
        private BoxCollider boxCollider;

        private void Start()
        {
            boxCollider = GetComponent<BoxCollider>();
        }

        private void LateUpdate()
        {

#if UNITY_EDITOR

            if (transform.rotation != Quaternion.identity)
            {
                Debug.LogError("Rotating fog null zone is not supported.");
            }

#endif

            if (WeatherMakerScript.Instance != null)
            {
                WeatherMakerScript.Instance.LightManagerScript.FogNullZones.Add(boxCollider.bounds);
            }

#if UNITY_EDITOR

            else
            {
                Debug.LogError("Unable to find WeatherMaker prefab in scene, unable to add fog null zone.");
            }

#endif

        }
    }
}