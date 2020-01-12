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
    public abstract class WeatherMakerFogScript : MonoBehaviour
    {
        #region Public fields

        [Header("Fog Appearance")]
        [Tooltip("Fog mode")]
        public WeatherMakerFogMode FogMode = WeatherMakerFogMode.Exponential;

        [Tooltip("Fog density")]
        [Range(0.0f, 1.0f)]
        public float FogDensity = 0.05f;

        [Tooltip("Fog color")]
        public Color FogColor = Color.white;

        [Tooltip("Whether to enable volumetric fog point/spot lights. Fog always uses directional lights. Disable to improve performance.")]
        public bool EnableFogLights = false;

        [Tooltip("Maximum fog factor, where 1 is the maximum allowed fog.")]
        [Range(0.0f, 1.0f)]
        public float MaxFogFactor = 1.0f;

        [Header("Fog Noise")]
        [Tooltip("Fog noise scale. Lower values get less tiling. 0 to disable noise.")]
        [Range(0.0f, 1.0f)]
        public float FogNoiseScale = 0.0001f;

        [Tooltip("Controls how the noise value is calculated. Negative values allow areas of no noise, higher values increase the intensity of the noise.")]
        [Range(-1.0f, 1.0f)]
        public float FogNoiseAdder = 0.0f;

        [Tooltip("How much the noise effects the fog.")]
        [Range(0.0f, 10.0f)]
        public float FogNoiseMultiplier = 0.15f;

        [Tooltip("Fog noise velocity, determines how fast the fog moves. Not all fog scripts support 3d velocity, some only support 2d velocity (x and y).")]
        public Vector3 FogNoiseVelocity = new Vector3(0.01f, 0.01f, 0.0f);
        private Vector3 fogNoiseVelocityAccum;

        [Tooltip("Number of samples to take for 3D fog. If the player will never enter the fog, this can be a lower value. If the player can move through the fog, 40 or higher is better, but will cost some performance.")]
        [Range(1, 100)]
        public int FogNoiseSampleCount = 40;

        [Header("Fog Rendering")]
        [Tooltip("Fog material")]
        public Material FogMaterial;

        [Tooltip("Dithering level. 0 to disable dithering.")]
        [Range(0.0f, 1.0f)]
        public float DitherLevel = 0.005f;

        [Header("Fog Shadows (Sun Only)")]
        [Tooltip("Number of shadow samples, 0 to disable fog shadows. Requires EnableFogLights be true.")]
        [Range(0, 100)]
        public int FogShadowSampleCount = 0;

        [Tooltip("Max ray length for fog shadows.")]
        [Range(10.0f, 1000.0f)]
        public float FogShadowMaxRayLength = 300.0f;

        [Tooltip("Multiplier for fog shadow lighting. Higher values make brighter light rays.")]
        [Range(0.0f, 32.0f)]
        public float FogShadowMultiplier = 8.0f;

        [Tooltip("Controls how light falls off from the light source. Higher values fall off faster. Setting this to a value that is a power of two is recommended.")]
        [Range(0.0f, 128.0f)]
        public float FogShadowPower = 64.0f;

        [Tooltip("Controls how light falls off from the light source. Lower values fall off faster.")]
        [Range(0.0f, 1.0f)]
        public float FogShadowDecay = 0.95f;

        [Tooltip("Fog shadow dither multiplier. Higher values dither more.")]
        [Range(0.0f, 3.0f)]
        public float FogShadowDither = 0.4f;

        [Tooltip("Magic numbers for fog shadow dithering. Tweak if you don't like the dithering appearance.")]
        public Vector4 FogShadowDitherMagic = new Vector4(0.73f, 1.665f, 1024.0f, 1024.0f);

        /// <summary>
        /// Density of fog for scattering reduction
        /// </summary>
        private float fogScatterReduction = 1.0f;
        public float FogScatterReduction { get { return fogScatterReduction; } }

        #endregion Public fields

        #region Public methods

        /// <summary>
        /// Set a new fog density over a period of time - if set to 0, game object will be disabled at end of transition
        /// </summary>
        /// <param name="fromDensity">Start of new fog density</param>
        /// <param name="toDensity">End of new fog density</param>
        /// <param name="transitionDuration">How long to transition to the new fog density in seconds</param>
        public void TransitionFogDensity(float fromDensity, float toDensity, float transitionDuration)
        {
            FogDensity = fromDensity;
            TweenFactory.Tween("WeatherMakerFog_" + gameObject.name, fromDensity, toDensity, transitionDuration, TweenScaleFunctions.Linear, (v) =>
            {
                FogDensity = v.CurrentValue;
            }, null);
        }

        public void SetFogShaderProperties(Material m)
        {
            bool gamma = (QualitySettings.activeColorSpace == ColorSpace.Gamma);
            float scatterCover = (WeatherMakerScript.Instance.CloudScript.enabled ? WeatherMakerScript.Instance.CloudScript.CloudCover : 0.0f);
            m.SetColor("_FogColor", FogColor);
            m.SetFloat("_FogNoiseScale", FogNoiseScale);
            m.SetFloat("_FogNoiseAdder", FogNoiseAdder);
            m.SetFloat("_FogNoiseMultiplier", FogNoiseMultiplier);
            m.SetVector("_FogNoiseVelocity", (fogNoiseVelocityAccum += (FogNoiseVelocity * Time.deltaTime * 0.005f)));
            m.SetFloat("_FogNoiseSampleCount", (float)FogNoiseSampleCount);
            m.SetFloat("_FogNoiseSampleCountInverse", 1.0f / (float)FogNoiseSampleCount);
            m.SetFloat("_MaxFogFactor", MaxFogFactor);
            if (m.IsKeywordEnabled("WEATHER_MAKER_FOG_CONSTANT"))
            {
                m.DisableKeyword("WEATHER_MAKER_FOG_CONSTANT");
            }
            else if (m.IsKeywordEnabled("WEATHER_MAKER_FOG_EXPONENTIAL"))
            {
                m.DisableKeyword("WEATHER_MAKER_FOG_EXPONENTIAL");
            }
            else if (m.IsKeywordEnabled("WEATHER_MAKER_FOG_LINEAR"))
            {
                m.DisableKeyword("WEATHER_MAKER_FOG_LINEAR");
            }
            else if (m.IsKeywordEnabled("WEATHER_MAKER_FOG_EXPONENTIAL_SQUARED"))
            {
                m.DisableKeyword("WEATHER_MAKER_FOG_EXPONENTIAL_SQUARED");
            }
            if (FogMode == WeatherMakerFogMode.None || FogDensity <= 0.0f || MaxFogFactor <= 0.001f)
            {
                fogScatterReduction = 1.0f;
            }
            else if (FogMode == WeatherMakerFogMode.Exponential)
            {
                m.EnableKeyword("WEATHER_MAKER_FOG_EXPONENTIAL");
                m.SetFloat("_FogDensityScatter", fogScatterReduction = Mathf.Clamp(1.0f - ((FogDensity + scatterCover) * 1.5f), 0.0f, 1.0f));
            }
            else if (FogMode == WeatherMakerFogMode.Linear)
            {
                m.EnableKeyword("WEATHER_MAKER_FOG_LINEAR");
                m.SetFloat("_FogDensityScatter", fogScatterReduction = Mathf.Clamp((1.0f - ((FogDensity + scatterCover) * 1.2f)), 0.0f, 1.0f));
            }
            else if (FogMode == WeatherMakerFogMode.ExponentialSquared)
            {
                m.EnableKeyword("WEATHER_MAKER_FOG_EXPONENTIAL_SQUARED");
                m.SetFloat("_FogDensityScatter", fogScatterReduction = Mathf.Clamp((1.0f - ((FogDensity + scatterCover) * 2.0f)), 0.0f, 1.0f));
            }
            else if (FogMode == WeatherMakerFogMode.Constant)
            {
                m.EnableKeyword("WEATHER_MAKER_FOG_CONSTANT");
                m.SetFloat("_FogDensityScatter", fogScatterReduction = Mathf.Clamp(1.0f - (FogDensity + scatterCover), 0.0f, 1.0f));
            }
            if (FogNoiseScale > 0.0f && FogNoiseMultiplier > 0.0f && WeatherMakerLightManagerScript.NoiseTexture3DInstance != null)
            {
                if (!m.IsKeywordEnabled("ENABLE_FOG_NOISE"))
                {
                    m.EnableKeyword("ENABLE_FOG_NOISE");
                }
            }
            else if (m.IsKeywordEnabled("ENABLE_FOG_NOISE"))
            {
                m.DisableKeyword("ENABLE_FOG_NOISE");
            }
            if (EnableFogLights)
            {
                if (FogShadowSampleCount > 0 && WeatherMakerScript.Instance.Sun.Light.intensity > 0.6f)
                {
                    float brightness = Mathf.Lerp(0.0f, 1.0f, (WeatherMakerScript.Instance.Sun.Light.intensity - 0.6f) / 0.4f);
                    m.DisableKeyword("ENABLE_FOG_LIGHTS");
                    m.EnableKeyword("ENABLE_FOG_LIGHTS_WITH_SHADOWS");
                    m.SetFloat("_FogLightShadowSampleCount", (float)FogShadowSampleCount);
                    m.SetFloat("_FogLightShadowInvSampleCount", 1.0f / (float)FogShadowSampleCount);
                    m.SetFloat("_FogLightShadowMaxRayLength", FogShadowMaxRayLength);
                    m.SetFloat("_FogLightShadowBrightness", brightness);
                    m.SetFloat("_FogLightShadowMultiplier", FogShadowMultiplier);
                    m.SetFloat("_FogShadowLightPower", FogShadowPower);
                    m.SetFloat("_FogShadowDecay", FogShadowDecay);
                    m.SetFloat("_FogLightShadowDither", FogShadowDither);
                    m.SetVector("_FogLightShadowDitherMagic", FogShadowDitherMagic);
                    if (QualitySettings.shadowCascades < 2)
                    {
                        m.EnableKeyword("SHADOWS_ONE_CASCADE");
                    }
                    else
                    {
                        m.DisableKeyword("SHADOWS_ONE_CASCADE");
                    }
                }
                else
                {
                    m.DisableKeyword("ENABLE_FOG_LIGHTS_WITH_SHADOWS");
                    m.EnableKeyword("ENABLE_FOG_LIGHTS");
                }
            }
            else
            {
                m.DisableKeyword("ENABLE_FOG_LIGHTS");
                m.DisableKeyword("ENABLE_FOG_LIGHTS_WITH_SHADOWS");
            }
            m.SetFloat("_FogDitherLevel", (gamma ? DitherLevel : DitherLevel * 0.5f));
            m.SetFloat("_FogDensity", FogDensity);
        }

        #endregion Public methods

        #region Protected methods

        protected virtual void Awake()
        {

#if UNITY_EDITOR

            if (Application.isPlaying)
            {

#endif

                // clone fog material
                FogMaterial = new Material(FogMaterial);

#if UNITY_EDITOR

            }

#endif

        }

        protected virtual void Start()
        {

#if UNITY_EDITOR

            if (WeatherMakerScript.Instance == null)
            {
                Debug.LogError("WeatherScript must be assigned on fog or exist in the scene");
            }

#endif

        }

        protected virtual void Update()
        {
            UpdateMaterial();
        }

        protected virtual void LateUpdate()
        {

        }

        protected virtual void OnDestroy()
        {
        }

        protected virtual void OnEnable()
        {
        }

        protected virtual void OnDisable()
        {
        }

        protected virtual void OnWillRenderObject()
        {
        }

        protected virtual void OnBecameVisible()
        {
        }

        protected virtual void OnBecameInvisible()
        {
        }

        protected virtual void UpdateMaterial()
        {

#if UNITY_EDITOR

            if (FogMaterial == null)
            {
                Debug.LogError("Must set fog material and fog blur material");
            }

#endif

            SetFogShaderProperties(FogMaterial);
        }

        #endregion Protected methods
    }
}
