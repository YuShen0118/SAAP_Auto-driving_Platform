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

using UnityEngine;
using System.Collections;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerWindScript : MonoBehaviour
    {
        [Tooltip("Wind intensity (0 - 1). MaximumWindSpeed * WindIntensity = WindSpeed.")]
        [Range(0.0f, 1.0f)]
        public float WindIntensity = 0.0f;

        [Tooltip("The absolute maximum of the wind speed. The wind zone wind main is set to WindIntensity * MaximumWindSpeed * WindMainMultiplier.")]
        [Range(0.0f, 1000.0f)]
        public float MaximumWindSpeed = 100.0f;

        [SingleLine("The maximum rotation the wind can change in degrees. For 2D, non-zero means random wind left or right.")]
        public RangeOfFloats WindMaximumChangeRotation = new RangeOfFloats { Minimum = 15.0f, Maximum = 60.0f };

        [Tooltip("Multiply the wind zone wind main by this value.")]
        [Range(0.0f, 1.0f)]
        public float WindMainMultiplier = 0.01f;

        [SingleLine("Wind turbulence range - set to a maximum 0 for no random turbulence.")]
        public RangeOfFloats WindTurbulenceRange = new RangeOfFloats { Minimum = 0.0f, Maximum = 100.0f };

        [SingleLine("Wind pulse magnitude range - set to a maximum of 0 for no random pulse magnitude.")]
        public RangeOfFloats WindPulseMagnitudeRange = new RangeOfFloats { Minimum = 2.0f, Maximum = 8.0f };

        [SingleLine("Wind pulse frequency range - set to a maximum of 0 for no random pulse frequency.")]
        public RangeOfFloats WindPulseFrequencyRange = new RangeOfFloats { Minimum = 0.01f, Maximum = 0.1f };

        [Tooltip("Whether random wind can blow upwards. Default is false.")]
        public bool AllowBlowUp = false;

        [Tooltip("Additional sound volume multiplier for the wind")]
        [Range(0.0f, 2.0f)]
        public float WindSoundMultiplier = 0.5f;

        [SingleLine("How often the wind speed and direction changes (minimum and maximum change interval in seconds). Set to 0 for no change.")]
        public RangeOfFloats WindChangeInterval = new RangeOfFloats { Minimum = 0.0f, Maximum = 30.0f };

        [Tooltip("How much the wind affects fog velocity")]
        [Range(0.0f, 1.0f)]
        public float FogVelocityMultiplier = 0.001f;

        /// <summary>
        /// The current wind velocity, not including turbulence and pulsing
        /// </summary>
        public Vector3 CurrentWindVelocity { get; private set; }

        /// <summary>
        /// Wind zone
        /// </summary>
        public WindZone WindZone { get; private set; }

        /// <summary>
        /// Wind audio source
        /// </summary>
        public WeatherMakerLoopingAudioSource AudioSourceWind { get; private set; }

        /// <summary>
        /// Allow notification of when the wind velocity changes
        /// </summary>
        public System.Action<Vector3> WindChanged { get; set; }

        /// <summary>
        /// Whether the wind direction is random. The wind direction is random if WindMaximumChangeRotation and WindChangeInterval are both greater than 0.
        /// </summary>
        public bool RandomWindDirection
        {
            get
            {
                return
                (
                    WindMaximumChangeRotation.Minimum > 0.0f &&
                    WindMaximumChangeRotation.Maximum >= WindMaximumChangeRotation.Minimum &&
                    WindChangeInterval.Minimum > 0.0f &&
                    WindChangeInterval.Maximum >= WindChangeInterval.Minimum
                );
            }
        }

        private Quaternion windChangeRotationStart;
        private Quaternion windChangeRotationEnd;
        private float windChangeElapsed;
        private float windChangeTotal;
        private float windNextChangeTime;
        private float lastWindIntensity = -1.0f;

        private void Awake()
        {
            WindZone = GetComponent<WindZone>();
            AudioSourceWind = new WeatherMakerLoopingAudioSource(GetComponent<AudioSource>());
        }

        private void UpdateWind()
        {
            // put wind on top of camera
            if (WeatherMakerScript.Instance.Camera != null)
            {
                WindZone.transform.position = WeatherMakerScript.Instance.Camera.transform.position;
                if (!WeatherMakerScript.Instance.CameraIsOrthographic)
                {
                    WindZone.transform.Translate(0.0f, WindZone.radius, 0.0f);
                }
            }

            if (WindIntensity > 0.0f && MaximumWindSpeed > 0.0f && WindMainMultiplier > 0.0f)
            {
                WindZone.windMain = MaximumWindSpeed * WindIntensity * WindMainMultiplier;

                // update wind audio if wind intensity changed
                if (WindZone.windMain != lastWindIntensity)
                {
                    AudioSourceWind.Play(WindIntensity * WindSoundMultiplier);
                }
                lastWindIntensity = WindZone.windMain;

                // check for wind change
                if (windNextChangeTime <= Time.time)
                {
                    // update to new wind
                    windNextChangeTime = Time.time + WindChangeInterval.Random();
                    windChangeTotal = 0.0f;
                    if (WindChangeInterval.Minimum > 0.0f && WindChangeInterval.Maximum >= WindChangeInterval.Minimum)
                    {
                        if (WindTurbulenceRange.Maximum > 0.0f)
                        {
                            WindZone.windTurbulence = WindTurbulenceRange.Random();
                        }
                        if (WindPulseMagnitudeRange.Maximum > 0.0f)
                        {
                            WindZone.windPulseMagnitude = WindPulseMagnitudeRange.Random();
                        }
                        if (WindPulseFrequencyRange.Maximum > 0.0f)
                        {
                            WindZone.windPulseFrequency = WindPulseFrequencyRange.Random();
                        }
                    }

                    // if random wind, pick a new direction from wind
                    if (RandomWindDirection)
                    {
                        // 2D is set immediately
                        if (WeatherMakerScript.Instance.CameraIsOrthographic)
                        {
                            int val = UnityEngine.Random.Range(0, 2);
                            WindZone.transform.rotation = Quaternion.Euler(0.0f, -90.0f + (180.0f * val), 0.0f);
                        }
                        // 3D is lerped over time
                        else
                        {
                            float xAxis = (AllowBlowUp ? UnityEngine.Random.Range(-30.0f, 30.0f) : 0.0f);
                            windChangeRotationStart = WindZone.transform.rotation;
                            windChangeRotationEnd = Quaternion.Euler(xAxis, UnityEngine.Random.Range(0.0f, 360.0f), 0.0f);
                            windChangeElapsed = 0.0f;
                            windChangeTotal = windNextChangeTime - Time.time;
                        }
                    }
                    // else use the WindZone transform rotation as is
                }
                // no change, update lerp for wind direction if needed
                else if (windChangeTotal != 0.0f)
                {
                    WindZone.transform.rotation = Quaternion.Lerp(windChangeRotationStart, windChangeRotationEnd, windChangeElapsed / windChangeTotal);
                    windChangeElapsed = Mathf.Min(windChangeTotal, windChangeElapsed + Time.deltaTime);
                }
                Vector3 newVelocity = WindZone.transform.forward * WindZone.windMain;
                bool velocityChanged = newVelocity != CurrentWindVelocity;
                CurrentWindVelocity = newVelocity;
                if (velocityChanged && WindChanged != null)
                {
                    WindChanged(newVelocity);
                }
                if (FogVelocityMultiplier != 0.0f && WeatherMakerScript.Instance.FogScript != null)
                {
                    WeatherMakerScript.Instance.FogScript.FogNoiseVelocity = CurrentWindVelocity * FogVelocityMultiplier;
                }
            }
            else
            {
                AudioSourceWind.Stop();
                WindZone.windMain = WindZone.windTurbulence = WindZone.windPulseFrequency = WindZone.windPulseMagnitude = 0.0f;
                CurrentWindVelocity = Vector3.zero;
            }
            AudioSourceWind.Update();
        }

        private void LateUpdate()
        {
            UpdateWind();
        }

        /// <summary>
        /// Animate wind intensity change
        /// </summary>
        /// <param name="value">New intensity</param>
        /// <param name="duration">Animation duration</param>
        public void AnimateWindIntensity(float value, float duration)
        {
            TweenFactory.Tween("WeatherMakerWindIntensity", WindIntensity, value, duration, TweenScaleFunctions.Linear, (t) => WindIntensity = t.CurrentValue, null);
        }
    }
}