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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    public static class WeatherMakerShaderIds
    {
        public static readonly int ArrayWeatherMakerMoonDirectionUp;
        public static readonly int ArrayWeatherMakerMoonDirectionDown;
        public static readonly int ArrayWeatherMakerMoonLightColor;
        public static readonly int ArrayWeatherMakerMoonLightPower;
        public static readonly int ArrayWeatherMakerMoonTintColor;
        public static readonly int ArrayWeatherMakerMoonTintIntensity;
        public static readonly int ArrayWeatherMakerMoonVar1;

        static WeatherMakerShaderIds()
        {
            ArrayWeatherMakerMoonDirectionUp = Shader.PropertyToID("_WeatherMakerMoonDirectionUp");
            ArrayWeatherMakerMoonDirectionDown = Shader.PropertyToID("_WeatherMakerMoonDirectionDown");
            ArrayWeatherMakerMoonLightColor = Shader.PropertyToID("_WeatherMakerMoonLightColor");
            ArrayWeatherMakerMoonLightPower = Shader.PropertyToID("_WeatherMakerMoonLightPower");
            ArrayWeatherMakerMoonTintColor = Shader.PropertyToID("_WeatherMakerMoonTintColor");
            ArrayWeatherMakerMoonTintIntensity = Shader.PropertyToID("_WeatherMakerMoonTintIntensity");
            ArrayWeatherMakerMoonVar1 = Shader.PropertyToID("_WeatherMakerMoonVar1");
        }

        public static void Initialize() { }
    }
}
