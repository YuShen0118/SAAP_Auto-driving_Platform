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
    public class WeatherMakerThunderAndLightningScript3D : WeatherMakerThunderAndLightningScript
    {
        [Header("3D settings")]
        [SingleLine("Range of distances away from the camera that normal lightning can be")]
        public RangeOfFloats NormalDistance = new RangeOfFloats { Minimum = 3000.0f, Maximum = 5000.0f };

        [SingleLine("Range of distances away from the camera that intense lightning can be")]
        public RangeOfFloats IntenseDistance = new RangeOfFloats { Minimum = 500.0f, Maximum = 1000.0f };

        protected override Vector3 CalculateStartPosition(ref Vector3 anchorPosition, Camera visibleInCamera, bool intense)
        {
            Vector3 start = anchorPosition;
            Vector3 randomDir;

            if (visibleInCamera == null)
            {
                randomDir = UnityEngine.Random.onUnitSphere;
            }
            else
            {
                Vector3 randomScreenPoint = new Vector3
                (
                    UnityEngine.Random.Range((float)visibleInCamera.pixelWidth * 0.3f, (float)visibleInCamera.pixelWidth * 0.7f),
                    UnityEngine.Random.Range((float)visibleInCamera.pixelHeight * 0.3f, (float)visibleInCamera.pixelHeight * 0.7f),
                    Random.Range(visibleInCamera.nearClipPlane, visibleInCamera.farClipPlane)
                );
                randomDir = (visibleInCamera.ScreenToWorldPoint(randomScreenPoint) - visibleInCamera.transform.position).normalized;
            }
            start += (randomDir * (intense ? IntenseDistance.Random() : NormalDistance.Random()));
            start.x += StartXVariance.Random();
            start.y = StartYVariance.Random() + StartYBase.Random();
            start.z += StartZVariance.Random();

            // if the start is too close to the anchor point, push it back
            float minDistance = (intense ? IntenseDistance.Minimum : NormalDistance.Minimum);
            if (Vector3.Distance(start, anchorPosition) < minDistance)
            {
                Vector3 startDir = (start - anchorPosition).normalized;
                start = anchorPosition + (startDir * minDistance);
            }

            return start;
        }

        protected override Vector3 CalculateEndPosition(ref Vector3 anchorPosition, ref Vector3 start, Camera visibleInCamera, bool intense)
        {
            Vector3 end = start;
            Vector3 dir;
            bool noGround = UnityEngine.Random.Range(0.0f, 1.0f) > GroundLightningChance;
            float minDistance = (intense ? IntenseDistance.Minimum : NormalDistance.Minimum);
            RaycastHit hit;

            // determine if we should strike the ground
            if (noGround)
            {
                end.y += EndYVariance.Random();
            }
            else if (Physics.Raycast(start, Vector3.down, out hit))
            {
                end.y = hit.point.y - 10.0f;
            }
            else
            {
                // strike ground, raycast will get actual ground point
                end.y = -10.0f;
            }

            if (visibleInCamera == null)
            {
                end.x = start.x + EndXVariance.Random();
                end.z = start.z + EndZVariance.Random();
            }
            else
            {
                end += (visibleInCamera.transform.right * EndXVariance.Random());
                end += (visibleInCamera.transform.forward * EndZVariance.Random());
            }

            dir = (end - start).normalized;

            // if the end is too close to the anchor point, push it back
            if (Vector3.Distance(anchorPosition, end) < minDistance || (visibleInCamera != null && Vector3.Dot(dir, visibleInCamera.transform.forward) > 0.1f))
            {
                if (visibleInCamera == null)
                {
                    dir = (end - anchorPosition).normalized;
                }
                else
                {
                    dir = visibleInCamera.transform.forward;
                }
                dir = dir.normalized;
                end = anchorPosition + (dir * minDistance);
            }

            // see if the bolt hit anything on it's way to the ground - if so, change the end point
            if (Physics.Raycast(start, dir, out hit, float.MaxValue))
            {
                end = hit.point;
            }

            return end;
        }

        protected override void Start()
        {
            base.Start();

#if DEBUG

            if (WeatherMakerScript.Instance.Camera == null)
            {
                Debug.LogWarning("Lightning requires a Camera be set on WeatherScript");
            }
            else if (WeatherMakerScript.Instance.Camera.farClipPlane < 2000.0f && !WeatherMakerScript.Instance.Camera.orthographic)
            {
                Debug.LogWarning("Far clip plane should be 2000+ for best lightning effects");
            }

#endif

        }

        protected override void Update()
        {
            base.Update();
        }
    }
}