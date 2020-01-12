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

#if GAIA_PRESENT && UNITY_EDITOR

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

using DigitalRuby.WeatherMaker;

namespace Gaia.GX.DigitalRuby
{
    [System.Serializable]
    public class WeatherMaker
    {
        private static WeatherMakerDayNightCycleScript SetLocation(float lat, float lon, bool setDirty = true)
        {
            if (WeatherMakerScript.Instance != null && WeatherMakerScript.Instance.DayNightScript != null && WeatherMakerScript.Instance.SkySphereScript != null)
            {
                WeatherMakerScript.Instance.DayNightScript.Latitude = lat;
                WeatherMakerScript.Instance.DayNightScript.Longitude = lon;
                WeatherMakerScript.Instance.DayNightScript.TimeZoneOffsetSeconds = -1111;
                if (setDirty)
                {
                    SerializationHelper.SetDirty(WeatherMakerScript.Instance.DayNightScript);
                }
                return WeatherMakerScript.Instance.DayNightScript;
            }
            return null;
        }

        private static WeatherMakerDayNightCycleScript SetDateTime(int year, int month, int day, int hour, int min, int sec, bool setDirty = true)
        {
            if (WeatherMakerScript.Instance != null && WeatherMakerScript.Instance.DayNightScript != null && WeatherMakerScript.Instance.SkySphereScript != null)
            {
                WeatherMakerScript.Instance.DayNightScript.Year = year;
                WeatherMakerScript.Instance.DayNightScript.Month = month;
                WeatherMakerScript.Instance.DayNightScript.Day = day;
                WeatherMakerScript.Instance.DayNightScript.TimeOfDay = (float)System.TimeSpan.FromSeconds((3600 * hour) + (60 * min) + sec).TotalSeconds;
                if (setDirty)
                {
                    SerializationHelper.SetDirty(WeatherMakerScript.Instance.DayNightScript);
                }
                return WeatherMakerScript.Instance.DayNightScript;
            }
            return null;
        }

        private static void SetTime(int hours, int minutes = 1, int seconds = 1)
        {
            SetTime(new System.TimeSpan(hours, minutes, seconds));
        }

        private static void SetTime(System.TimeSpan t)
        {
            if (WeatherMakerScript.Instance != null && WeatherMakerScript.Instance.DayNightScript != null && WeatherMakerScript.Instance.SkySphereScript != null)
            {
                WeatherMakerScript.Instance.DayNightScript.TimeOfDay = (float)t.TotalSeconds;
                SerializationHelper.SetDirty(WeatherMakerScript.Instance.DayNightScript);
            }
        }

        private static void SetMonth(int month)
        {
            SetDateTime(System.DateTime.Now.Year, month, 15, 11, 1, 1);
        }

        public static string GetPublisherName()
        {
            return "Digital Ruby, LLC";
        }

        public static string GetPackageName()
        {
            return "Weather Maker";
        }

        public static void GX_About()
        {
            UnityEditor.EditorUtility.DisplayDialog("About Weather Maker",
@"
Weather Maker is your all in one solution for Weather, Sky and Volumetric Fog. With 2D and 3D versions,
Weather Maker is ready for any type of game or app.

I'm ready to help with any questions, please email me at support@digitalruby.com and I'll do my best to help.

- Jeff Johnson (http://www.digitalruby.com)
", "OK");
        }

        public static void GX_SetTimeOfDay_Dawn()
        {
            if (WeatherMakerScript.Instance != null && WeatherMakerScript.Instance.DayNightScript != null && WeatherMakerScript.Instance.SkySphereScript != null)
            {
                SetTime(WeatherMakerScript.Instance.DayNightScript.SunData.Dawn);
            }
        }

        public static void GX_SetTimeOfDay_SunRise()
        {
            if (WeatherMakerScript.Instance != null && WeatherMakerScript.Instance.DayNightScript != null && WeatherMakerScript.Instance.SkySphereScript != null)
            {
                SetTime(WeatherMakerScript.Instance.DayNightScript.SunData.SunRise);
            }
        }

        public static void GX_SetTimeOfDay_Morning()
        {
            SetTime(9);
        }

        public static void GX_SetTimeOfDay_Noon()
        {
            SetTime(12);
        }

        public static void GX_SetTimeOfDay_Afternoon()
        {
            SetTime(15);
        }

        public static void GX_SetTimeOfDay_Evening()
        {
            SetTime(18);
        }

        public static void GX_SetTimeOfDay_Sunset()
        {
            if (WeatherMakerScript.Instance != null && WeatherMakerScript.Instance.DayNightScript != null && WeatherMakerScript.Instance.SkySphereScript != null)
            {
                SetTime(WeatherMakerScript.Instance.DayNightScript.SunData.SunSet);
            }
        }

        public static void GX_SetTimeOfDay_Dusk()
        {
            if (WeatherMakerScript.Instance != null && WeatherMakerScript.Instance.DayNightScript != null && WeatherMakerScript.Instance.SkySphereScript != null)
            {
                SetTime(WeatherMakerScript.Instance.DayNightScript.SunData.Dusk);
            }
        }

        public static void GX_SetTimeOfDay_Night()
        {
            SetTime(1);
        }

        public static void GX_SetMonth_January()
        {
            SetMonth(1);
        }

        public static void GX_SetMonth_February()
        {
            SetMonth(2);
        }

        public static void GX_SetMonth_March()
        {
            SetMonth(3);
        }

        public static void GX_SetMonth_April()
        {
            SetMonth(4);
        }

        public static void GX_SetMonth_May()
        {
            SetMonth(5);
        }

        public static void GX_SetMonth_June()
        {
            SetMonth(6);
        }

        public static void GX_SetMonth_July()
        {
            SetMonth(7);
        }

        public static void GX_SetMonth_August()
        {
            SetMonth(8);
        }

        public static void GX_SetMonth_September()
        {
            SetMonth(9);
        }

        public static void GX_SetMonth_October()
        {
            SetMonth(10);
        }

        public static void GX_SetMonth_November()
        {
            SetMonth(11);
        }

        public static void GX_SetMonth_December()
        {
            SetMonth(12);
        }

        public static void GX_SetLocation_London()
        {
            SetLocation(51.5074f, -0.1278f);
        }

        public static void GX_SetLocation_NewYork()
        {
            SetLocation(40.7128f, -74.0059f);
        }

        public static void GX_SetLocation_MexicoCity()
        {
            SetLocation(19.4326f, 99.1332f);
        }

        public static void GX_SetLocation_Tokyo()
        {
            SetLocation(35.6895f, 139.6917f);
        }

        public static void GX_SetLocation_Paris()
        {
            SetLocation(48.8566f, 2.3522f);
        }

        public static void GX_SetLocation_Moscow()
        {
            SetLocation(55.7558f, 37.6173f);
        }

        public static void GX_SetLocation_Sydney()
        {
            SetLocation(-33.8688f, 151.2093f);
        }

        public static void GX_SetLocation_SãoPaulo()
        {
            SetLocation(23.5505f, 46.6333f);
        }

        public static void GX_SetLocation_NorthPole()
        {
            SetLocation(90.0f, 0.0f);
        }

        public static void GX_SetLocation_SouthPole()
        {
            SetLocation(-90f, 0.0f);
        }

        public static void GX_SetMoon_Full()
        {
            SetLocation(40.2338f, -111.6585f, false);
            WeatherMakerDayNightCycleScript d = SetDateTime(2017, 4, 11, 23, 57, 1, false);
            SerializationHelper.SetDirty(d);
        }

        public static void GX_SetMoon_ThirdQuarter()
        {
            SetLocation(40.2338f, -111.6585f, false);
            WeatherMakerDayNightCycleScript d = SetDateTime(2017, 4, 7, 3, 57, 1, false);
            SerializationHelper.SetDirty(d);
        }

        public static void GX_SetMoon_SecondQuarter()
        {
            SetLocation(40.2338f, -111.6585f, false);
            WeatherMakerDayNightCycleScript d = SetDateTime(2017, 4, 4, 23, 57, 1, false);
            SerializationHelper.SetDirty(d);
        }

        public static void GX_SetMoon_FirstQuarter()
        {
            SetLocation(40.2338f, -111.6585f, false);
            WeatherMakerDayNightCycleScript d = SetDateTime(2017, 4, 1, 23, 57, 1, false);
            SerializationHelper.SetDirty(d);
        }

        public static void GX_SetMoon_New()
        {
            SetLocation(40.2338f, -111.6585f, false);
            WeatherMakerDayNightCycleScript d = SetDateTime(2009, 3, 26, 23, 57, 1, false);
            SerializationHelper.SetDirty(d);
        }

        public static void GX_SetMoon_TotalSolarEclipse()
        {
            SetLocation(43.8231f, -111.7924f, false);
            WeatherMakerDayNightCycleScript script = SetDateTime(2017, 8, 21, 10, 13, 0, false);
            SerializationHelper.SetDirty(script);
        }
    }
}

#endif
