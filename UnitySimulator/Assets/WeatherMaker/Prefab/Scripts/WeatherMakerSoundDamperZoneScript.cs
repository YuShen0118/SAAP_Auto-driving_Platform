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
    [RequireComponent(typeof(Collider))]
    public class WeatherMakerSoundDamperZoneScript : MonoBehaviour
    {
        [Tooltip("The multiplier for Weather Maker sounds if the zone is entered.")]
        [Range(0.0f, 1.0f)]
        public float Dampening = 0.2f;

        [Tooltip("Transition duration in seconds.")]
        [Range(0.0f, 10.0f)]
        public float TransitionDuration = 3.0f;

        [Tooltip("Required tag for the other collider. Generally this will be Player but you can set it to something else.")]
        public string RequiredTag = "Player";

        private Collider _collider;
        private static int triggers;

        private float CurrentSoundDampening()
        {
            float val;
            if (!WeatherMakerScript.Instance.VolumeModifierDictionary.TryGetValue("WeatherMakerSoundDamperZoneScript", out val))
            {
                val = 1.0f;
            }
            return val;
        }

        private void Start()
        {
            _collider = GetComponent<Collider>();

#if UNITY_EDITOR

            if (!_collider.isTrigger)
            {
                Debug.LogError("WeatherMakerSoundDamperZoneScript only works with trigger colliders.");
            }

#endif

            UnityEngine.SceneManagement.SceneManager.sceneUnloaded += SceneManager_sceneUnloaded;
        }

        private void SceneManager_sceneUnloaded(UnityEngine.SceneManagement.Scene arg0)
        {
            triggers = 0;
        }

        private void OnTriggerEnter(Collider other)
        {
            // if this is the first trigger entered, run it
            if (other.gameObject.tag == RequiredTag && ++triggers == 1)
            {
                TweenFactory.Tween("WeatherMakerSoundDamperZoneScript", CurrentSoundDampening(), Dampening, TransitionDuration, TweenScaleFunctions.Linear, (t) =>
                {
                    WeatherMakerScript.Instance.VolumeModifierDictionary["WeatherMakerSoundDamperZoneScript"] = t.CurrentValue;
                });
            }
        }

        private void OnTriggerExit(Collider other)
        {
            // if this is the last trigger exited, run it
            if (other.gameObject.tag == RequiredTag && --triggers == 0)
            {
                TweenFactory.Tween("WeatherMakerSoundDamperZoneScript", CurrentSoundDampening(), 1.0f, TransitionDuration, TweenScaleFunctions.Linear, (t) =>
                {
                    WeatherMakerScript.Instance.VolumeModifierDictionary["WeatherMakerSoundDamperZoneScript"] = t.CurrentValue;
                }, (t) =>
                {
                    WeatherMakerScript.Instance.VolumeModifierDictionary.Remove("WeatherMakerSoundDamperZoneScript");
                });
            }
        }
    }
}