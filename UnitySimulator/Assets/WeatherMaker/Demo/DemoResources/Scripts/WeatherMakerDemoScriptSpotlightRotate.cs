using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerDemoScriptSpotlightRotate : MonoBehaviour
    {
        private float currentDuration;
        private float totalDuration;
        private Quaternion startRotation;
        private Quaternion endRotation;

        private void Start()
        {
        }

        private void Update()
        {
            currentDuration -= Time.deltaTime;
            if (currentDuration <= 0.0f)
            {
                totalDuration = currentDuration = UnityEngine.Random.Range(3.0f, 6.0f);
                Vector3 ray = UnityEngine.Random.insideUnitSphere;
                ray.y = Mathf.Min(ray.y, -0.25f);
                startRotation = transform.rotation;
                endRotation = Quaternion.LookRotation(ray);
            }
            transform.rotation = Quaternion.Lerp(startRotation, endRotation, 1.0f - (currentDuration / totalDuration));
        }
    }
}