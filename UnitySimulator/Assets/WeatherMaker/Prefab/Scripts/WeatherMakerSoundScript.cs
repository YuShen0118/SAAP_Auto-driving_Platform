// #define ENABLE_DEBUG_LOG_WEATHER_MAKER_SOUNDS

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    [CreateAssetMenu(fileName = "WeatherMakerSound", menuName = "WeatherMaker/Sound", order = 2)]
    public class WeatherMakerSoundScript : WeatherMakerBaseScriptableObjectScript
    {
        private float timeToNextPlay;
        private float timeRemainingToPlay;

        [Tooltip("A name to help you keep track of the ambient sound")]
        public string Name;

        [Tooltip("The audio clip to play (one is chosen at random)")]
        public AudioClip[] AudioClips;

        [Tooltip("Whether to loop the audio clip, all clips must be looping or not looping.")]
        public bool Looping;

        [Tooltip("The times of day the sound can play. In order to play HoursOfDay must be empty or must also match.")]
        [EnumFlag]
        public WeatherMakerTimeOfDayCategory TimesOfDay;

        [Tooltip("The hours in a day the sound can play, or null/empty to just use TimesOfDay.")]
        public List<RangeOfFloats> HoursOfDay;

        [SingleLine("How long in seconds to wait in between playing the sound. Set to 0 min and max to have no wait and a continous sound.")]
        public RangeOfFloats IntervalRange;

        [SingleLine("How long in seconds to play the sound, for non-looped AudioSource this is ignored, and the entire sound is played.")]
        public RangeOfFloats DurationRange;

        [SingleLine("Range of volume for the ambient sound, a new value is chosen each time the sound is played.")]
        public RangeOfFloats VolumeRange = new RangeOfFloats { Minimum = 1.0f, Maximum = 1.0f };

        [SingleLine("How long in seconds to fade in and out for looping audio sources.")]
        public RangeOfFloats FadeDuration = new RangeOfFloats { Minimum = 5.0f, Maximum = 15.0f };

        /// <summary>
        /// Wrapper for AudioSource to help with looping
        /// </summary>
        public WeatherMakerLoopingAudioSource AudioSourceLoop { get; set; }

        /// <summary>
        /// Whether the sound can play - this is usually true unless something like an audio zone is used and the player is not in the zone.
        /// </summary>
        public bool CanPlay { get; set; }

        public override void Awake()
        {
            base.Awake();

            CanPlay = true;

#if UNITY_EDITOR

            if (!Application.isPlaying)
            {
                return;
            }

#endif

            if (AudioSourceLoop == null)
            {
                WeatherMakerScript.AssertExists();
                AudioSource source = WeatherMakerScript.Instance.gameObject.AddComponent<AudioSource>();
                source.loop = Looping;
                source.playOnAwake = false;
                source.hideFlags = HideFlags.HideAndDontSave | HideFlags.HideInInspector;
                AudioSourceLoop = new WeatherMakerLoopingAudioSource(source, 0.0f, 0.0f, Looping);
            }
        }

        public void Stop()
        {

#if UNITY_EDITOR

            if (!Application.isPlaying || AudioSourceLoop == null)
            {
                return;
            }

#endif

            if (AudioSourceLoop.AudioSource.isPlaying && !AudioSourceLoop.Stopping)
            {

#if ENABLE_DEBUG_LOG_WEATHER_MAKER_SOUNDS

                Debug.LogFormat("Weather Maker stopping sound {0}", Name);

#endif

                AudioSourceLoop.Stop();
                timeToNextPlay = 0.0f;
            }
        }

        public override void Update()
        {

#if UNITY_EDITOR

            if (!Application.isPlaying || WeatherMakerScript.Instance == null)
            {
                return;
            }
            else

#endif

            if (AudioSourceLoop == null)
            {
                return;
            }
            else if (AudioSourceLoop.AudioSource.isPlaying)
            {
                // see if we need to stop
                if (!AudioSourceLoop.Stopping && ((timeRemainingToPlay -= Time.deltaTime) < 0.0f && IntervalRange.Minimum > 0.0f) || !CanStartSound())
                {
                    Stop();
                }
            }
            // check if it is the right time of day to play the ambient sound
            else if (CanStartSound())
            {
                if (timeToNextPlay <= 0.0f && IntervalRange.Minimum > 0.0f)
                {
                    CalculateNextPlay();
                }
                // check if it is time to play
                else if ((timeToNextPlay -= Time.deltaTime) < 0.0f)
                {
                    timeToNextPlay = 0.0f;
                    timeRemainingToPlay = (!Looping || DurationRange.Maximum <= 0.0f ? float.MaxValue : DurationRange.Random());
                    float startFade = Looping ? FadeDuration.Random() : 0.0f;
                    float endFade = Looping ? FadeDuration.Random() : 0.0f;
                    AudioSourceLoop.SetFade(startFade, endFade);
                    AudioSourceLoop.AudioSource.clip = AudioClips[UnityEngine.Random.Range(0, AudioClips.Length)];
                    AudioSourceLoop.Play(VolumeRange.Random());

#if ENABLE_DEBUG_LOG_WEATHER_MAKER_SOUNDS

                    if (Looping)
                    {
                        Debug.LogFormat("Weather Maker playing sound {0} for {1} seconds", Name, (timeRemainingToPlay == float.MaxValue ? "Infinite" : timeRemainingToPlay.ToString("0.00")));
                    }
                    else
                    {
                        Debug.LogFormat("Weather Maker playing sound {0} for {1:0.00} seconds", Name, AudioClip.length);
                    }

#endif

                }
            }
            else
            {
                timeToNextPlay = 0.0f;
            }
            AudioSourceLoop.VolumeModifier = WeatherMakerScript.Instance.cachedVolumeModifier;
            AudioSourceLoop.Update();
        }

        protected override void OnDisable()
        {
            base.OnDisable();
            Stop();
        }

        protected override void OnDestroy()
        {
            base.OnDestroy();
            if (AudioSourceLoop != null)
            {
                GameObject.Destroy(AudioSourceLoop.AudioSource);
            }
        }

        private bool CanStartSound()
        {
            if (!Disabled && CanPlay && AudioClips != null && AudioClips.Length != 0 && (int)(WeatherMakerScript.Instance.DayNightScript.TimeOfDayCategory & TimesOfDay) != 0)
            {
                if (HoursOfDay.Count == 0)
                {
                    return true;
                }
                foreach (RangeOfFloats hours in HoursOfDay)
                {
                    if (WeatherMakerScript.Instance.DayNightScript.TimeOfDayTimespan.TotalHours >= hours.Minimum && WeatherMakerScript.Instance.DayNightScript.TimeOfDayTimespan.TotalHours <= hours.Maximum)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        private void CalculateNextPlay(float delay = 0.0f)
        {
            timeToNextPlay = IntervalRange.Random() + delay;

#if ENABLE_DEBUG_LOG_WEATHER_MAKER_SOUNDS

            Debug.LogFormat("Weather Maker sound {0} will play in {1:0.00} seconds", Name, timeToNextPlay);

#endif

        }
    }
}