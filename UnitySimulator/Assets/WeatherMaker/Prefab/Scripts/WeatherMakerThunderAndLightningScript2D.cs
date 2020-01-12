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
using System;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerThunderAndLightningScript2D : WeatherMakerThunderAndLightningScript
    {
        private void CalculateVisibleBounds(out Vector3 visibleMin, out Vector3 visibleMax, out float width, out float height)
        {
            if (WeatherMakerScript.Instance.Camera == null)
            {
                Debug.LogError("Unable to calculate visible bounds, Camera property must be set on WeatherMakerScript");
            }
            visibleMin = WeatherMakerScript.Instance.Camera.ViewportToWorldPoint(Vector3.zero);
            visibleMax = WeatherMakerScript.Instance.Camera.ViewportToWorldPoint(Vector3.one);
            width = (visibleMax.x - visibleMin.x);
            height = (visibleMax.y - visibleMin.y);
        }

        protected override Vector3 CalculateStartPosition(ref Vector3 anchorPosition, Camera visibleInCamera, bool intense)
        {
            Vector3 visibleMin, visibleMax;
            float width, height;
            CalculateVisibleBounds(out visibleMin, out visibleMax, out width, out height);
            Vector3 start = new Vector3(visibleMin.x + (width * UnityEngine.Random.Range(0.2f, 0.8f)), visibleMin.y + (height * StartYBase.Random()), 0.0f);
            start.x += (width * StartXVariance.Random());
            start.y += (height * StartYVariance.Random());

            return start;
        }

        protected override Vector3 CalculateEndPosition(ref Vector3 anchorPosition, ref Vector3 start, Camera visibleInCamera, bool intense)
        {
            Vector3 dir = UnityEngine.Random.insideUnitSphere;
            dir.y = -Mathf.Abs(dir.y);
            dir.z = 0.0f;
            dir = dir.normalized;

            RaycastHit2D h;
            if (UnityEngine.Random.Range(0.0f, 1.0f) >= GroundLightningChance ||
                ((h = Physics2D.Raycast(start, dir)).collider == null))
            {
                Vector3 visibleMin, visibleMax;
                float width, height;
                CalculateVisibleBounds(out visibleMin, out visibleMax, out width, out height);
                float maxDimen = Mathf.Max(width, height);
                float variance = Mathf.Max(maxDimen * 0.25f, maxDimen * UnityEngine.Random.Range(EndYVariance.Minimum, EndYVariance.Maximum));
                Vector3 end = start + (dir * variance);
                end.x += (width * EndXVariance.Random());
                end.y += (height * EndYVariance.Random());
                end.z = 0.0f;

                return end;
            }

            return h.point;
        }
    }
}