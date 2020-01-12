using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    [CreateAssetMenu(fileName = "WeatherMakerSoundGroup", menuName = "WeatherMaker/Sound Group", order = 3)]
    [System.Serializable]
    public class WeatherMakerSoundGroupScript : WeatherMakerBaseScriptableObjectScript
    {
        [Tooltip("All sounds in the group")]
        public List<WeatherMakerSoundScript> Sounds;

        public bool CanPlay
        {
            get
            {
                return Sounds == null || Sounds.Count == 0 || Sounds[0] == null ? false : Sounds[0].CanPlay;
            }
            set
            {
                if (Sounds != null)
                {
                    foreach (WeatherMakerSoundScript script in Sounds)
                    {
                        if (script != null)
                        {
                            script.CanPlay = value;
                        }
                    }
                }
            }
        }

        public override void Awake()
        {
            base.Awake();
            if (Sounds != null)
            {
                foreach (WeatherMakerSoundScript sound in Sounds)
                {
                    if (sound != null)
                    {
                        sound.Awake();
                    }
                }
            }
        }

        public override void Update()
        {
            base.Update();
            if (Sounds != null)
            {
                foreach (WeatherMakerSoundScript sound in Sounds)
                {
                    if (sound != null)
                    {
                        sound.Update();
                    }
                }
            }
        }
    }
}