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
    [CreateAssetMenu(fileName = "WeatherMakerProfile", menuName = "WeatherMaker/Profile", order = 1)]
    public class WeatherMakerProfileScript : WeatherMakerBaseScriptableObjectScript
    {
        [SingleLine("Random range in seconds this profile takes to transition in. This will affect how long the clouds, wind, precipitation and fog take to transition.")]
        [Range(0.0f, 300.0f)]
        public RangeOfFloats TransitionDuration = new RangeOfFloats { Minimum = 20.0f, Maximum = 60.0f };

        [Header("Clouds")]
        [Tooltip("Type of clouds.")]
        public WeatherMakerCloudType CloudType;

        [Tooltip("Color of clouds.")]
        public Color CloudColor = Color.white;

        [SingleLine("Cloud speed random range.")]
        public RangeOfFloats CloudSpeed;


        [Header("Precipitation")]
        [Tooltip("Type of precipitation.")]
        public WeatherMakerPrecipitationType Precipitation;

        [Tooltip("Precipitation intensity, 0 to 1.")]
        [Range(0.0f, 1.0f)]
        public float PrecipitationIntensity;


        [Header("Fog")]
        [Tooltip("Density of the fog, 0 to 1.")]
        [Range(0.0f, 1.0f)]
        public float FogDensity;

        [Tooltip("Fog height. Set to 0 for unlimited height.")]
        [Range(0.0f, 5000.0f)]
        public float FogHeight;

        [Header("Wind")]
        [Tooltip("Intensity of the wind, 0 to 1.")]
        [Range(0.0f, 1.0f)]
        public float WindIntensity;

        [Tooltip("The absolute maximum of the wind speed. The wind zone wind main is set to WindIntensity * MaximumWindSpeed * WindMainMultiplier.")]
        [Range(0.0f, 1000.0f)]
        public float MaximumWindSpeed = 100.0f;

        [SingleLine("The maximum rotation the wind can change in degrees. For 2D, non-zero means random wind left or right.")]
        public RangeOfFloats WindMaximumChangeRotation = new RangeOfFloats { Minimum = 15.0f, Maximum = 60.0f };

        [Tooltip("Multiply the wind zone wind main by this value.")]
        [Range(0.0f, 1.0f)]
        public float WindMainMultiplier = 0.01f;


        [Header("Lightning")]
        [Tooltip("Whether lightning is enabled")]
        public bool LightningEnabled;

        [Tooltip("Probability lightning will be intense (close up and loud).")]
        [Range(0.0f, 1.0f)]
        public float LightningIntenseProbability;

        [SingleLine("The random range of seconds in between lightning strikes.")]
        public RangeOfFloats LightningIntervalTimeRange = new RangeOfFloats { Minimum = 10.0f, Maximum = 25.0f };

        [Tooltip("Probability that lightning strikes will be forced to be visible in the camera view. Even if this fails, there is still " +
            "a change that the lightning will be visible. Ignored for some modes such as 2D.")]
        [Range(0.0f, 1.0f)]
        public float LightningForcedVisibilityProbability = 0.5f;

        [Tooltip("The chance that non-cloud lightning will hit the ground.")]
        [Range(0.0f, 1.0f)]
        public float LightningGroundChance = 0.3f;

        [Tooltip("The chance lightning will simply be in the clouds with no visible bolt.")]
        [Range(0.0f, 1.0f)]
        public float LightningCloudChance = 0.5f;


        [Header("Sounds")]
        [Tooltip("Possible sounds that can play.")]
        public WeatherMakerSoundGroupScript[] Sounds;
    }
}
