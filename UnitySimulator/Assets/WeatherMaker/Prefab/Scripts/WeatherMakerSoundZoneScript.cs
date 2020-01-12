using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerSoundZoneScript : MonoBehaviour
    {
        [Tooltip("Sounds to play if in the sound zone")]
        public List<WeatherMakerSoundGroupScript> Sounds;

        private void OnTriggerEnter(Collider other)
        {
            // entered the trigger, can play
            foreach (WeatherMakerSoundGroupScript script in Sounds)
            {
                script.CanPlay = true;
            }
        }

        private void OnTriggerExit(Collider other)
        {
            foreach (WeatherMakerSoundGroupScript script in Sounds)
            {
                // left the trigger, can't play
                script.CanPlay = false;
            }
        }

        private void Start()
        {
            foreach (WeatherMakerSoundGroupScript script in Sounds)
            {
                script.Awake();

                // can't play until we enter the trigger
                script.CanPlay = false;
            }
        }

        private void LateUpdate()
        {
            // update all sounds
            foreach (WeatherMakerSoundGroupScript script in Sounds)
            {
                script.Update();
            }
        }
    }
}