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
    /// Integration for AQUAS water and river set
    /// </summary>
    [ExecuteInEditMode]
    public class WeatherMakerExtensionAquasScript : WeatherMakerExtensionWaterScript

#if AQUAS_PRESENT

        <AQUAS_Reflection>

#else

        <UnityEngine.MonoBehaviour>

#endif

    {

#if AQUAS_PRESENT

        private AQUAS_LensEffects lensScript;
        private bool fogWasEnabled;
        private bool wasUnderwater;

        protected override void Awake()
        {
            base.Awake();

            lensScript = GameObject.FindObjectOfType<AQUAS_LensEffects>();
        }

        protected override void LateUpdate()
        {
            base.LateUpdate();

            if (WeatherMakerScript.Instance.CloudScript != null && TypeScript != null)
            {
                Renderer renderer = TypeScript.GetComponent<Renderer>();
                if (renderer != null)
                {
                    float specular = Mathf.Pow(WeatherMakerScript.Instance.FogScript.FogScatterReduction, CloudCoverWaterSpecularPower);
                    specular = Mathf.Min(specular, Mathf.Pow(1.0f - WeatherMakerScript.Instance.CloudScript.CloudCover, CloudCoverWaterSpecularPower));
                    float reflection = Mathf.Pow(WeatherMakerScript.Instance.FogScript.FogScatterReduction, CloudCoverWaterReflectionPower);
                    reflection = Mathf.Min(reflection, Mathf.Pow(1.0f - WeatherMakerScript.Instance.CloudScript.CloudCover, CloudCoverWaterReflectionPower));
                    reflection = Mathf.Min(0.5f, reflection);
                    renderer.sharedMaterial.SetFloat("_Specular", specular);
                    renderer.sharedMaterial.SetFloat("_ReflectionIntensity", reflection);
                }
                if (WeatherMakerScript.Instance.FogScript.FogDensity > 0.0f && lensScript != null)
                {
                    if (lensScript.underWater)
                    {
                        if (!wasUnderwater)
                        {
                            wasUnderwater = true;
                            fogWasEnabled = WeatherMakerScript.Instance.FogScript.enabled;
                        }
                        WeatherMakerScript.Instance.FogScript.enabled = false;
                    }
                    else if (wasUnderwater)
                    {
                        WeatherMakerScript.Instance.FogScript.enabled = fogWasEnabled;
                        fogWasEnabled = wasUnderwater = false;
                    }
                }
            }
        }

#endif

    }
}