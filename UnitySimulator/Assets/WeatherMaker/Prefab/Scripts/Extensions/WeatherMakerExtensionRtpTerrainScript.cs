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
    /// <summary>
    /// Integration for Uber - Standard Shader Ultra
    /// </summary>
    [ExecuteInEditMode]
    public class WeatherMakerExtensionRtpTerrainScript : WeatherMakerExtensionRainSnowSeasonScript

#if RELIEF_TERRAIN_PACK_PRESENT

        <ReliefTerrain>

#else

        <UnityEngine.MonoBehaviour>

#endif

    {

#if RELIEF_TERRAIN_PACK_PRESENT

        [Tooltip("The minimum water level.")]
        [Range(0.0f, 1.0f)]
        public float MinWaterLevel = 0.0f;

        [Tooltip("The minimum wetness amount.")]
        [Range(0.0f, 1.0f)]
        public float MinWetnessAmount = 0.0f;

        [Tooltip("The minimum snow amount.")]
        [Range(0.0f, 1.0f)]
        public float MinSnowAmount = 0.0f;

        protected override void OnUpdateRain(float rain)
        {
            TypeScript.globalSettingsHolder.TERRAIN_RainIntensity = rain;
            TypeScript.globalSettingsHolder.TERRAIN_GlobalWetness = Mathf.Max(MinWetnessAmount, rain);
            Shader.SetGlobalFloat("TERRAIN_RainIntensity", rain);
            Shader.SetGlobalFloat("TERRAIN_GlobalWetness", TERRAIN_GlobalWetness);
        }

        protected override void OnUpdateSnow(float snow)
        {
            TypeScript.globalSettingsHolder._snow_strength = Mathf.Max(MinSnowAmount, snow);
            Shader.SetGlobalFloat("rtp_snow_strength", _snow_strength);
        }

#endif

    }
}
