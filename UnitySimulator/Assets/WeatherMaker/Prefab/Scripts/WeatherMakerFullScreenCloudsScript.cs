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
using UnityEngine.Rendering;

using System;
using System.Collections.Generic;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerFullScreenCloudsScript : MonoBehaviour
    {
        [Header("Full Screen Clouds - Rendering")]
        [Tooltip("Cloud shadow projector script")]
        public WeatherMakerCloudShadowProjectorScript CloudShadowProjectorScript;

        [Tooltip("Cloud rendering material.")]
        public Material Material;

        [Tooltip("Material to blit the full screen clouds.")]
        public Material FullScreenMaterial;

        [Tooltip("Down sample scale.")]
        [Range(0.01f, 1.0f)]
        public float DownSampleScale = 1.0f;

        [Tooltip("Blur Material.")]
        public Material BlurMaterial;

        [Tooltip("Blur Shader Type.")]
        public BlurShaderType BlurShader;

        [Tooltip("Render Queue")]
        public CameraEvent RenderQueue = CameraEvent.BeforeImageEffectsOpaque;

        [Header("Clouds - Noise")]
        [Tooltip("Texture for cloud noise - r noise, g = normal (0-1), b = noise2, a = normal (0-1). R channel only used currently.")]
        public Texture2D CloudNoise;

        [Tooltip("Cloud noise scale")]
        [Range(0.000001f, 1.0f)]
        public float CloudNoiseScale = 0.02f;

        [Tooltip("Multiplier for cloud noise")]
        [Range(0.0f, 4.0f)]
        public float CloudNoiseMultiplier = 1.0f;

        [Tooltip("Cloud noise velocity (xz)")]
        public Vector2 CloudNoiseVelocity;
        private Vector2 cloudNoiseVelocityAccum;

        [Tooltip("Texture for masking cloud noise, makes clouds visible in only certain parts of the sky.")]
        public Texture2D CloudNoiseMask;

        [Tooltip("Cloud noise mask scale")]
        [Range(0.000001f, 1.0f)]
        public float CloudNoiseMaskScale = 0.5f;

        [Tooltip("Cloud noise mask rotation in degrees")]
        [Range(0.0f, 360.0f)]
        public float CloudNoiseMaskRotation = 0.0f;

        [Tooltip("Offset for cloud noise mask")]
        public Vector2 CloudNoiseMaskOffset;

        [Tooltip("Cloud noise mask velocity (xz)")]
        public Vector2 CloudNoiseMaskVelocity;
        private Vector2 cloudNoiseMaskVelocityAccum;

        [Header("Clouds - Appearance")]
        [Tooltip("Cloud color, for lighting")]
        public Color CloudColor = Color.white;

        [Tooltip("Cloud emission color, always emits this color regardless of lighting.")]
        public Color CloudEmissionColor = Color.clear;

        [Tooltip("Cloud height - only affects where the clouds stop at the horizon")]
        public float CloudHeight = 500;

        [Tooltip("Cloud cover, controls how many clouds there are")]
        [Range(0.0f, 1.0f)]
        public float CloudCover = 0.2f;

        [Tooltip("Cloud density, controls how opaque the clouds are")]
        [Range(0.0f, 1.0f)]
        public float CloudDensity = 0.0f;

        [Tooltip("Cloud light absorption. As this approaches 0, all light is absorbed.")]
        [Range(0.0f, 1.0f)]
        public float CloudLightAbsorption = 0.013f;

        [Tooltip("Cloud sharpness, controls how distinct the clouds are")]
        [Range(0.0f, 1.0f)]
        public float CloudSharpness = 0.015f;

        [Tooltip("Cloud whispiness, controls how thin / small particles the clouds get as they change over time.")]
        [Range(0.0f, 3.0f)]
        public float CloudWhispiness = 1.0f;

        [Tooltip("Changes the whispiness of the clouds over time")]
        [Range(0.0f, 1.0f)]
        public float CloudWhispinessChangeFactor = 0.03f;

        [Tooltip("Cloud pixels with alpha greater than this will cast a shadow. Set to 1 to disable cloud shadows.")]
        [Range(0.0f, 1.0f)]
        public float CloudShadowThreshold = 0.1f;

        [Tooltip("Cloud shadow power. 0 is full power, higher loses power.")]
        [Range(0.0001f, 2.0f)]
        public float CloudShadowPower = 0.5f;

        [Tooltip("Bring clouds down at the horizon at the cost of stretching over the top.")]
        [Range(0.0f, 0.5f)]
        public float CloudRayOffset = 0.1f;

        [Tooltip("Used to block lens flare if clouds are over the sun. Just needs to be a sphere collider.")]
        public GameObject CloudLensFlareBlocker;

        private WeatherMakerFullScreenEffect effect;

        private void UpdateShaderProperties(WeatherMakerCommandBuffer b)
        {
            SetShaderCloudParameters(b.Material);
        }

        private void Start()
        {
            effect = new WeatherMakerFullScreenEffect
            {
                CommandBufferName = "WeatherMakerFullScreenCloudsScript",
                DownsampleRenderBufferTextureName = "_MainTex2",
                RenderQueue = RenderQueue,
                ZTest = CompareFunction.LessEqual
            };
        }

        private void UpdateLensFlare(Camera c)
        {
            if (WeatherMakerScript.Instance.Sun == null)
            {
                return;
            }
            LensFlare flare = WeatherMakerScript.Instance.Sun.Transform.GetComponent<LensFlare>();
            if (flare == null)
            {
                return;
            }
            if (CloudCover < 0.5f)
            {
                CloudLensFlareBlocker.SetActive(false);
            }
            else
            {
                CloudLensFlareBlocker.SetActive(true);
                Vector3 toSun = (c.transform.position - WeatherMakerScript.Instance.Sun.Transform.position).normalized;
                CloudLensFlareBlocker.transform.position = c.transform.position + (toSun * 16.0f);
            }
        }

        private void Update()
        {

        }

        private void LateUpdate()
        {
            effect.Material = Material;
            effect.BlitMaterial = FullScreenMaterial;
            effect.BlurMaterial = BlurMaterial;
            effect.BlurShaderType = BlurShader;
            effect.DownsampleScale = DownSampleScale;
            effect.DownsampleRenderBufferScale = 0.0f;
            effect.UpdateMaterialProperties = UpdateShaderProperties;
            effect.Enabled = CloudCover > Mathf.Epsilon;
            effect.LateUpdate();
            foreach (WeatherMakerCelestialObject obj in WeatherMakerScript.Instance.Suns)
            {
                obj.Renderer.enabled = (CloudCover < 0.85f || CloudNoiseMask != null);
            }
            foreach (WeatherMakerCelestialObject obj in WeatherMakerScript.Instance.Moons)
            {
                obj.Renderer.enabled = (CloudCover < 0.85f || CloudNoiseMask != null);
            }
        }

        private void OnDestroy()
        {
            if (effect != null)
            {
                effect.Dispose();
            }
        }

        private void OnDisable()
        {
            if (effect != null)
            {
                effect.Dispose();
            }
        }

        internal void SetShaderCloudParameters(Material m)
        {
            m.DisableKeyword("ENABLE_CLOUDS");
            m.DisableKeyword("ENABLE_CLOUDS_MASK");

            if (CloudsEnabled)
            {
                m.SetTexture("_FogNoise", CloudNoise);
                m.SetColor("_FogColor", CloudColor);
                m.SetColor("_FogEmissionColor", CloudEmissionColor);
                m.SetVector("_FogNoiseVelocity", (cloudNoiseVelocityAccum += (CloudNoiseVelocity * Time.deltaTime * 0.005f)));
                if (CloudNoiseMask == null)
                {
                    m.EnableKeyword("ENABLE_CLOUDS");
                }
                else
                {
                    m.EnableKeyword("ENABLE_CLOUDS_MASK");
                    m.SetTexture("_FogNoiseMask", CloudNoiseMask);
                    m.SetVector("_FogNoiseMaskOffset", CloudNoiseMaskOffset);
                    m.SetVector("_FogNoiseMaskVelocity", (cloudNoiseMaskVelocityAccum += (CloudNoiseMaskVelocity * Time.deltaTime * 0.005f)));
                    m.SetFloat("_FogNoiseMaskScale", CloudNoiseMaskScale * 0.01f);
                    float rotRadians = CloudNoiseMaskRotation * Mathf.Deg2Rad;
                    m.SetFloat("_FogNoiseMaskRotationSin", Mathf.Sin(rotRadians));
                    m.SetFloat("_FogNoiseMaskRotationCos", Mathf.Cos(rotRadians));
                }
                m.SetFloat("_FogNoiseScale", CloudNoiseScale);
                m.SetFloat("_FogNoiseMultiplier", CloudNoiseMultiplier);
                m.SetFloat("_FogHeight", CloudHeight);
                m.SetFloat("_FogCover", CloudCover);
                m.SetFloat("_FogDensity", CloudDensity);
                m.SetFloat("_FogLightAbsorption", CloudLightAbsorption);
                m.SetFloat("_FogSharpness", CloudSharpness);
                m.SetFloat("_FogWhispiness", CloudWhispiness);
                m.SetFloat("_FogWhispinessChangeFactor", CloudWhispinessChangeFactor);
                m.SetFloat("_FogShadowThreshold", CloudShadowThreshold);
                float shadowDotPower = Mathf.Clamp(Mathf.Pow(3.0f * Vector3.Dot(Vector3.down, WeatherMakerScript.Instance.Sun.Transform.forward), 0.5f), 0.0f, 1.0f);
                float shadowPower = Mathf.Lerp(2.0f, CloudShadowPower, shadowDotPower);
                m.SetFloat("_FogShadowPower", shadowPower);
                m.SetFloat("_WeatherMakerCloudRayOffset", CloudRayOffset);

#if UNITY_EDITOR

                if (Application.isPlaying)
                {

#endif

                    float cover = CloudCover * (1.5f - CloudLightAbsorption);
                    float sunIntensityMultiplier = Mathf.Clamp(1.0f - (CloudDensity * 0.5f), 0.0f, 1.0f);
                    WeatherMakerScript.Instance.DayNightScript.DirectionalLightIntensityMultipliers["WeatherMakerSkySphereScript"] = sunIntensityMultiplier;
                    float sunShadowMultiplier = Mathf.Lerp(1.0f, 0.0f, Mathf.Clamp(((CloudDensity + cover) * 0.85f), 0.0f, 1.0f));
                    WeatherMakerScript.Instance.DayNightScript.DirectionalLightShadowIntensityMultipliers["WeatherMakerSkySphereScript"] = sunShadowMultiplier;

#if UNITY_EDITOR

                }

#endif

            }
            else
            {

#if UNITY_EDITOR

                if (Application.isPlaying)
                {

#endif

                    WeatherMakerScript.Instance.DayNightScript.DirectionalLightIntensityMultipliers["WeatherMakerFullScreenCloudsScript"] = 1.0f;
                    WeatherMakerScript.Instance.DayNightScript.DirectionalLightShadowIntensityMultipliers["WeatherMakerFullScreenCloudsScript"] = 1.0f;

#if UNITY_EDITOR

                }

#endif

            }
        }

        public void PreCullCamera(Camera camera)
        {
            if (effect != null)
            {
                effect.SetupCamera(camera);
            }

#if UNITY_EDITOR

            if (Application.isPlaying)
            {

#endif

                UpdateLensFlare(camera);
                CloudShadowProjectorScript.RenderCloudShadows(camera);

#if UNITY_EDITOR

            }

#endif

        }

        /// <summary>
        /// Checks whether clouds are enabled
        /// </summary>
        public bool CloudsEnabled
        {
            get { return (CloudNoise != null && CloudColor.a > 0.0f && CloudNoiseMultiplier > 0.0f && CloudCover > 0.0f); }
        }

        /// <summary>
        /// Show cloud animated
        /// </summary>
        /// <param name="duration">How long until clouds fully transition to the parameters</param>
        /// <param name="cover">Cloud cover, 0 to 1</param>
        /// <param name="density">Cloud density, 0 to 1</param>
        /// <param name="whispiness">Cloud whispiness, 0 to 3</param>
        /// <param name="sharpness">Cloud sharpness, controls fade, 0 to 1</param>
        /// <param name="lightAbsorption">Cloud light absoprtion, 0 to 1, higher values absorb less</param>
        /// <param name="color">Cloud color, if null defaults to CloudColor</param>
        public void ShowCloudsAnimated(float duration, float cover, float density = -1.0f, float whispiness = -1.0f, float sharpness = -1.0f, float lightAbsorption = -1.0f, Color? color = null)
        {
            float startCover = CloudCover;
            float startDensity = CloudDensity;
            float startWhispiness = CloudWhispiness;
            float startSharpness = CloudSharpness;
            float startLightAbsorption = CloudLightAbsorption;
            Color startColor = CloudColor;
            color = color ?? CloudColor;
            density = (density < 0.0f ? CloudDensity : density);
            whispiness = (whispiness < 0.0f ? CloudWhispiness : whispiness);
            sharpness = (sharpness < 0.0f ? CloudSharpness : sharpness);
            lightAbsorption = (lightAbsorption < 0.0f ? CloudLightAbsorption : lightAbsorption);
            TweenFactory.Tween("WeatherMakerClouds", 0.0f, 1.0f, duration, TweenScaleFunctions.Linear, (ITween<float> c) =>
            {
                CloudCover = Mathf.Lerp(startCover, cover, c.CurrentValue);
                CloudDensity = Mathf.Lerp(startDensity, density, c.CurrentValue);
                CloudWhispiness = Mathf.Lerp(startWhispiness, whispiness, c.CurrentValue);
                CloudColor = Color.Lerp(startColor, color.Value, c.CurrentValue);
                CloudSharpness = Mathf.Lerp(startSharpness, sharpness, c.CurrentValue);
                CloudLightAbsorption = Mathf.Lerp(startLightAbsorption, lightAbsorption, c.CurrentValue);
            });
        }

        public void HideCloudsAnimated(float duration)
        {
            float cover = CloudCover;
            float density = CloudDensity;
            TweenFactory.Tween("WeatherMakerClouds", 1.0f, 0.0f, duration, TweenScaleFunctions.Linear, (ITween<float> c) =>
            {
                CloudCover = c.CurrentValue * cover;
                CloudDensity = c.CurrentValue * density;
            });
        }
    }
}
