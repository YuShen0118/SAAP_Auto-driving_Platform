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
    public enum WeatherMakerFogMode
    {
        None,
        Constant,
        Linear,
        Exponential,
        ExponentialSquared
    }

    public class WeatherMakerFullScreenFogScript : WeatherMakerFogScript
    {
        [Header("Full Screen Fog - Rendering")]
        [Tooltip("Material to render the fog full screen after it has been calculated")]
        public Material FogFullScreenMaterial;

        [Tooltip("Down sample scale.")]
        [Range(0.01f, 1.0f)]
        public float DownSampleScale = 1.0f;

        [Tooltip("Fog Blur Material.")]
        public Material FogBlurMaterial;

        [Tooltip("Fog Blur Shader Type.")]
        public BlurShaderType BlurShader;

        [Tooltip("Fog height. Set to 0 for unlimited height.")]
        [Range(0.0f, 5000.0f)]
        public float FogHeight = 0.0f;

        [Tooltip("Depth buffer less than far plane multiplied by this value will occlude the sun light through the fog.")]
        [Range(0.0f, 1.0f)]
        public float FarPlaneSunThreshold = 0.75f;

        [Tooltip("Render fog in this render queue for the command buffer.")]
        public CameraEvent FogRenderQueue = CameraEvent.BeforeForwardAlpha;

        [Header("Full Screen Fog - Screen Space Sun Shafts")]
        [Tooltip("The number of sun shaft samples. Set to 0 to disable sun shafts.")]
        [Range(0, 100)]
        public int SunShaftSampleCount = 0;

        [Tooltip("Sun shaft down sample scale. Down samples camera buffer to this before rendering fog.")]
        [Range(0.01f, 1.0f)]
        public float SunShaftDownSampleScale = 0.5f;

        [Tooltip("Sun shaft spread (0 - 1).")]
        [Range(0.0f, 1.0f)]
        public float SunShaftSpread = 0.65f;

        [Tooltip("Increases the sun shaft brightness")]
        [Range(0.0f, 1.0f)]
        public float SunShaftBrightness = 0.075f;

        [Tooltip("Combined with each ray march, this determines how much light is accumulated each step.")]
        [Range(0.0f, 100.0f)]
        public float SunShaftStepMultiplier = 21.0f;

        [Tooltip("Determines light fall-off from start of sun shaft. Set to 1 for no fall-off.")]
        [Range(0.5f, 1.0f)]
        public float SunShaftDecay = 0.97f;

        [Tooltip("Sun shaft tint color. Alpha value determines tint intensity.")]
        public Color SunShaftTintColor = Color.white;

        [Tooltip("Controls dithering intensity of sun shafts.")]
        [Range(-1.0f, 1.0f)]
        public float SunShaftDither = 0.4f;

        [Tooltip("Controls dithering appearance of sun shafts.")]
        public Vector4 SunShaftDitherMagic = new Vector4(2.34325f, 5.235345f, 1024.0f, 1024.0f);

        private WeatherMakerFullScreenEffect effect;

        private const string commandBufferName = "WeatherMakerFullScreenFogScript";

        private void UpdateFogProperties()
        {
            float multiplier;
            if (FogMode == WeatherMakerFogMode.Constant || FogMode == WeatherMakerFogMode.Linear)
            {
                float h = (FogHeight <= 0.0f ? 1000.0f : FogHeight) * 0.01f;
                multiplier = 1.0f - (FogDensity * 4.0f * h);
            }
            else if (FogMode == WeatherMakerFogMode.Exponential)
            {
                float h = (FogHeight <= 0.0f ? 1000.0f : FogHeight) * 0.02f;
                multiplier = 1.0f - Mathf.Min(1.0f, Mathf.Pow(FogDensity * 32.0f * h, 0.5f));
            }
            else
            {
                float h = (FogHeight <= 0.0f ? 1000.0f : FogHeight) * 0.04f;
                multiplier = 1.0f - Mathf.Min(1.0f, Mathf.Pow(FogDensity * 64.0f * h, 0.5f));
            }
            WeatherMakerScript.Instance.DayNightScript.DirectionalLightShadowIntensityMultipliers["WeatherMakerFullScreenFogScript"] = Mathf.Clamp(multiplier, 0.0f, 1.0f);
            effect.Material = FogMaterial;
            effect.BlitMaterial = FogFullScreenMaterial;
            effect.BlurMaterial = FogBlurMaterial;
            effect.BlurShaderType = BlurShader;
            effect.DownsampleScale = DownSampleScale;
            effect.DownsampleRenderBufferScale = (SunShaftSampleCount <= 0 ? 0.0f : SunShaftDownSampleScale);
            effect.UpdateMaterialProperties = UpdateShaderProperties;
            effect.Enabled = (FogDensity > Mathf.Epsilon && FogMode != WeatherMakerFogMode.None);
            effect.LateUpdate();
        }

        private void UpdateShaderProperties(WeatherMakerCommandBuffer b)
        {
            SetFogShaderProperties(b.Material);
            if (FogHeight > 0.0f)
            {
                b.Material.SetFloat("_FogHeight", FogHeight);
                if (!b.Material.IsKeywordEnabled("ENABLE_FOG_HEIGHT"))
                {
                    b.Material.EnableKeyword("ENABLE_FOG_HEIGHT");
                }
            }
            else if (b.Material.IsKeywordEnabled("ENABLE_FOG_HEIGHT"))
            {
                b.Material.DisableKeyword("ENABLE_FOG_HEIGHT");
            }

            // if no sun, then no sun shafts
            if (WeatherMakerScript.Instance.Sun == null)
            {
                return;
            }
            else if (SunShaftSampleCount > 0 && WeatherMakerScript.Instance.Sun.ViewportPosition.z < 0.0f)
            {
                if (!b.Material.IsKeywordEnabled("ENABLE_FOG_SUN_SHAFTS"))
                {
                    // Sun is visible
                    b.Material.EnableKeyword("ENABLE_FOG_SUN_SHAFTS");
                }

                // as sun leaves viewport or goes below horizon, fade out the shafts
                Vector2 viewportCenter = new Vector2((b.Camera.rect.min.x + b.Camera.rect.max.x) * 0.5f, (b.Camera.rect.min.y + b.Camera.rect.max.y) * 0.5f);
                float sunDistanceFromCenterViewport = ((Vector2)WeatherMakerScript.Instance.Sun.ViewportPosition - viewportCenter).magnitude * 0.5f;
                float sunDistanceFromHorizon = Mathf.Max(0.0f, -WeatherMakerScript.Instance.Sun.Transform.forward.y);
                float sunFadeOut = Mathf.SmoothStep(1.0f, 0.0f, sunDistanceFromCenterViewport) * sunDistanceFromHorizon;
                bool gamma = (QualitySettings.activeColorSpace == ColorSpace.Gamma);
                float brightness = SunShaftBrightness * (gamma ? 1.0f : 0.33f) * WeatherMakerScript.Instance.Sun.Light.intensity * sunFadeOut * Mathf.Max(0.0f, 1.0f - (WeatherMakerScript.Instance.CloudScript.CloudCover * 1.5f));
                b.Material.SetVector("_FogSunShaftsParam1", new Vector4(SunShaftSpread / (float)SunShaftSampleCount, (float)SunShaftSampleCount, brightness, 1.0f / (float)SunShaftSampleCount));
                b.Material.SetVector("_FogSunShaftsParam2", new Vector4(SunShaftStepMultiplier, SunShaftDecay, SunShaftDither, 0.0f));
                b.Material.SetVector("_FogSunShaftsTintColor", new Vector4(SunShaftTintColor.r * SunShaftTintColor.a, SunShaftTintColor.g * SunShaftTintColor.a,
                    SunShaftTintColor.b * SunShaftTintColor.a, SunShaftTintColor.a));
                b.Material.SetVector("_FogSunShaftsDitherMagic", SunShaftDitherMagic);
            }
            else if (b.Material.IsKeywordEnabled("ENABLE_FOG_SUN_SHAFTS"))
            {
                // disable sun shafts
                b.Material.DisableKeyword("ENABLE_FOG_SUN_SHAFTS");
            }
        }

        protected override void Start()
        {
            base.Start();

            effect = new WeatherMakerFullScreenEffect
            {
                CommandBufferName = commandBufferName,
                DownsampleRenderBufferTextureName = "_MainTex2",
                RenderQueue = FogRenderQueue
            };

#if UNITY_EDITOR

            if (Application.isPlaying)
            {

#endif

                FogBlurMaterial = new Material(FogBlurMaterial);

#if UNITY_EDITOR

            }

#endif

        }

        protected override void LateUpdate()
        {
            base.LateUpdate();
            UpdateFogProperties();
        }

        protected override void OnDestroy()
        {
            base.OnDestroy();
            if (effect != null)
            {
                effect.Dispose();
            }
        }

        protected override void OnDisable()
        {
            base.OnDisable();
            if (effect != null)
            {
                effect.Dispose();
            }
        }

        public void PreCullCamera(Camera camera)
        {
            if (effect != null)
            {
                effect.SetupCamera(camera);
            }
        }
    }
}