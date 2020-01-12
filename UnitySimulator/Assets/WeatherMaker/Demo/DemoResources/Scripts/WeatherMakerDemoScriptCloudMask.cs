using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerDemoScriptCloudMask : MonoBehaviour
    {
        private void Start()
        {
            WeatherMakerScript.Instance.ConfigurationScript.CloudDropdown.value = 3;
        }

        private void Update()
        {

        }
    }
}
