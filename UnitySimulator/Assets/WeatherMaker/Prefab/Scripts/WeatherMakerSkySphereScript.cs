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
using System.Collections;

namespace DigitalRuby.WeatherMaker
{
    /// <summary>
    /// Sky modes
    /// </summary>
    public enum WeatherMakeSkyMode
    {
        /// <summary>
        /// Textured - day, dawn/dusk and night are all done via textures
        /// </summary>
        Textured = 0,

        /// <summary>
        /// Procedural sky - day and dawn/dusk textures are overlaid on top of procedural sky, night texture is used as is
        /// </summary>
        ProceduralTextured,

        /// <summary>
        /// Procedural sky - day, dawn/dusk textures are ignored, night texture is used as is
        /// </summary>
        Procedural
    }

    [ExecuteInEditMode]
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
    public class WeatherMakerSkySphereScript : WeatherMakerSphereCreatorScript
    {
        [Header("Sky Rendering")]
        [Tooltip("The sky mode. 'Textured' uses a texture for day, dawn/dusk and night. " +
            "'Procedural textured' combines a procedural sky with the day and dawn/dusk textures using alpha, and uses the night texture as is. " +
            "'Procedural' uses the night texture as is and does everything else procedurally.")]
        public WeatherMakeSkyMode SkyMode = WeatherMakeSkyMode.Textured;

        [Range(0.0f, 1.0f)]
        [Tooltip("Dither level")]
        public float DitherLevel = 0.005f;

        [Header("Positioning")]
        [Range(-0.5f, 0.5f)]
        [Tooltip("Offset the sky this amount from the camera y. This value is multiplied by the height of the sky sphere.")]
        public float YOffsetMultiplier = 0.0f;

        [Range(0.1f, 1.0f)]
        [Tooltip("Place the sky sphere at this amount of the far clip plane")]
        public float FarClipPlaneMultiplier = 0.8f;

        [Header("Textures - dawn/dusk not used in procedural sky.")]
        [Tooltip("The daytime texture")]
        public Texture2D DayTexture;

        [Tooltip("The dawn / dusk texture (not used for procedural sky) - this MUST be set if DawnDuskFadeDegrees is not 0, otherwise things will look funny.")]
        public Texture2D DawnDuskTexture;

        [Tooltip("The night time texture")]
        public Texture2D NightTexture;

        [Header("Night Sky")]
        [Range(0.0f, 1.0f)]
        [Tooltip("Night pixels must have an R, G or B value greater than or equal to this to be visible. Raise this value if you want to hide dimmer elements " +
            "of your night texture or there is a lot of light pollution, i.e. a city.")]
        public float NightVisibilityThreshold = 0.0f;

        [Range(0.0f, 32.0f)]
        [Tooltip("Intensity of night sky. Pixels that don't meet the NightVisibilityThreshold will still be invisible.")]
        public float NightIntensity = 2.0f;

        [Range(0.0f, 100.0f)]
        [Tooltip("How fast the twinkle pulsates")]
        public float NightTwinkleSpeed = 16.0f;

        [Tooltip("The variance of the night twinkle. The higher the value, the more variance.")]
        [Range(0.0f, 10.0f)]
        public float NightTwinkleVariance = 1.0f;

        [Tooltip("The minimum of the max rgb component for the night pixel to twinkle")]
        [Range(0.0f, 1.0f)]
        public float NightTwinkleMinimum = 0.02f;

        [Tooltip("The amount of randomness in the night sky twinkle")]
        [Range(0.0f, 5.0f)]
        public float NightTwinkleRandomness = 0.15f;

        [Header("Sky Parameters (Advanced)")]
        [Tooltip("Allows eclipses - beware Unity bug that causes raycast to be very expensive. If you see CPU spike, disable.")]
        public bool CheckForEclipse;

        [Range(0.0f, 1.0f)]
        [Tooltip("Sky camera height (km)")]
        public float SkyCameraHeight = 0.0001f;

        [Tooltip("Sky tint color")]
        public Color SkyTintColor = new Color(0.5f, 0.5f, 0.5f);

        [Range(0.0f, 10.0f)]
        [Tooltip("Sky atmosphere mie, controls glow around the sun, etc.")]
        public float SkyAtmosphereMie = 0.99f;

        [Range(0.0f, 5.0f)]
        [Tooltip("Sky atmosphere thickness")]
        public float SkyAtmosphereThickness = 1.0f;

        [Range(1.0f, 2.0f)]
        [Tooltip("Sky outer radius")]
        public float SkyOuterRadius = 1.025f;

        [Range(1.0f, 2.0f)]
        [Tooltip("Sky inner radius")]
        public float SkyInnerRadius = 1.0f;

        [Range(0.0f, 500.0f)]
        [Tooltip("Sky mie multiplier")]
        public float SkyMieMultiplier = 1.0f;

        [Range(0.0f, 100.0f)]
        [Tooltip("Sky rayleigh multiplier")]
        public float SkyRayleighMultiplier = 1.0f;

        // light wave length constants
        internal const float lightWaveLengthRed = 0.65f;
        internal const float lightWaveLengthGreen = 0.570f;
        internal const float lightWaveLengthBlue = 0.475f;
        internal const float lightWaveTintRange = 0.15f;

        private RaycastHit[] eclipseHits = new RaycastHit[16];

        private void SetBlendMode(BlendMode src, BlendMode dst)
        {
            Material.SetInt("_SrcBlendMode", (int)src);
            Material.SetInt("_DstBlendMode", (int)dst);
        }

        private void SetShaderSkyParameters()
        {
            if (Material == null)
            {
                return;
            }
            Material.mainTexture = DayTexture;
            Material.SetTexture("_DawnDuskTex", DawnDuskTexture);
            Material.SetTexture("_NightTex", NightTexture);
            Material.DisableKeyword("ENABLE_TEXTURED_SKY");
            Material.DisableKeyword("ENABLE_PROCEDURAL_SKY");
            Material.DisableKeyword("ENABLE_PROCEDURAL_TEXTURED_SKY");
            if (SkyMode == WeatherMakeSkyMode.Textured)
            {
                Material.EnableKeyword("ENABLE_TEXTURED_SKY");
                SetBlendMode(BlendMode.One, BlendMode.Zero);
            }
            else if (SkyMode == WeatherMakeSkyMode.Procedural)
            {
                Material.EnableKeyword("ENABLE_PROCEDURAL_SKY");
                SetBlendMode(BlendMode.One, BlendMode.Zero);
            }
            else if (SkyMode == WeatherMakeSkyMode.ProceduralTextured)
            {
                Material.EnableKeyword("ENABLE_PROCEDURAL_TEXTURED_SKY");
                SetBlendMode(BlendMode.One, BlendMode.Zero);
            }

            SetGlobalSkyParameters(SkyAtmosphereMie, SkyMieMultiplier, SkyRayleighMultiplier, SkyAtmosphereThickness,
                SkyTintColor, SkyOuterRadius, SkyInnerRadius, SkyCameraHeight);

#if DEBUG

            if (WeatherMakerScript.Instance.CurrentCamera == null)
            {
                Debug.LogWarning("Sky sphere requires a camera be set on WeatherScript");
            }
            else

#endif

            {
                bool gamma = (QualitySettings.activeColorSpace == ColorSpace.Gamma);
                float radius = (WeatherMakerScript.Instance.CurrentCamera.farClipPlane * WeatherMakerScript.Instance.SkySphereScript.FarClipPlaneMultiplier) * 0.95f;
                Shader.SetGlobalFloat("_WeatherMakerSkySphereRadius", radius);
                Shader.SetGlobalFloat("_WeatherMakerSkySphereRadiusSquared", radius * radius);
                Shader.SetGlobalFloat("_WeatherMakerSkyDitherLevel", (gamma ? DitherLevel : DitherLevel * 0.5f));
            }
        }

        internal static void SetGlobalSkyParameters(float skyAtmosphereMie, float skyMieMultiplier, float skyRayleighMultiplier, float skyAtmosphereThickness,
            Color skyTintColor, float skyOuterRadius, float skyInnerRadius, float skyCameraHeight)
        {
            // global sky parameters
            float mieG = -skyAtmosphereMie;
            float mieG2 = skyAtmosphereMie * skyAtmosphereMie;
            float mieConstant = 0.001f * skyMieMultiplier;
            float rayleighConstant = 0.0025f * skyRayleighMultiplier;
            rayleighConstant = Mathf.LerpUnclamped(0.0f, rayleighConstant, Mathf.Pow(skyAtmosphereThickness, 2.5f));
            float lightWaveLengthRedTint = Mathf.Lerp(lightWaveLengthRed - lightWaveTintRange, lightWaveLengthRed + lightWaveTintRange, 1.0f - skyTintColor.r);
            float lightWaveLengthGreenTint = Mathf.Lerp(lightWaveLengthGreen - lightWaveTintRange, lightWaveLengthGreen + lightWaveTintRange, 1.0f - skyTintColor.g);
            float lightWaveLengthBlueTint = Mathf.Lerp(lightWaveLengthBlue - lightWaveTintRange, lightWaveLengthBlue + lightWaveTintRange, 1.0f - skyTintColor.b);
            float lightWaveLengthRed4 = lightWaveLengthRedTint * lightWaveLengthRedTint * lightWaveLengthRedTint * lightWaveLengthRedTint;
            float lightWaveLengthGreen4 = lightWaveLengthGreenTint * lightWaveLengthGreenTint * lightWaveLengthGreenTint * lightWaveLengthGreenTint;
            float lightWaveLengthBlue4 = lightWaveLengthBlueTint * lightWaveLengthBlueTint * lightWaveLengthBlueTint * lightWaveLengthBlueTint;
            float lightInverseWaveLengthRed4 = 1.0f / lightWaveLengthRed4;
            float lightInverseWaveLengthGreen4 = 1.0f / lightWaveLengthGreen4;
            float lightInverseWaveLengthBlue4 = 1.0f / lightWaveLengthBlue4;
            const float sunBrightnessFactor = 40.0f;
            float sunRed = rayleighConstant * sunBrightnessFactor * lightInverseWaveLengthRed4;
            float sunGreen = rayleighConstant * sunBrightnessFactor * lightInverseWaveLengthGreen4;
            float sunBlue = rayleighConstant * sunBrightnessFactor * lightInverseWaveLengthBlue4;
            float sunIntensity = mieConstant * sunBrightnessFactor;
            float pi4Red = rayleighConstant * 4.0f * Mathf.PI * lightInverseWaveLengthRed4;
            float pi4Green = rayleighConstant * 4.0f * Mathf.PI * lightInverseWaveLengthGreen4;
            float pi4Blue = rayleighConstant * 4.0f * Mathf.PI * lightInverseWaveLengthBlue4;
            float pi4Intensity = mieConstant * 4.0f * Mathf.PI;
            float scaleFactor = 1.0f / (skyOuterRadius - 1.0f);
            const float scaleDepth = 0.25f;
            float scaleOverScaleDepth = scaleFactor / scaleDepth;
            if (WeatherMakerScript.Instance != null)
            {
                float sunGradientLookup = (WeatherMakerScript.Instance.Camera.orthographic ? WeatherMakerScript.Instance.Sun.Transform.forward.z : sunGradientLookup = -WeatherMakerScript.Instance.Sun.Transform.forward.y);
                sunGradientLookup = ((sunGradientLookup + 1.0f) * 0.5f);
                Shader.SetGlobalColor("_WeatherMakerSkyGradientColor", WeatherMakerScript.Instance.DayNightScript.SkyGradient.Evaluate(sunGradientLookup));
            }
            Shader.SetGlobalFloat("_WeatherMakerSkySamples", 3.0f);
            Shader.SetGlobalFloat("_WeatherMakerSkyMieG", mieG);
            Shader.SetGlobalFloat("_WeatherMakerSkyMieG2", mieG2);
            Shader.SetGlobalFloat("_WeatherMakerSkyAtmosphereThickness", skyAtmosphereThickness);
            Shader.SetGlobalVector("_WeatherMakerSkyRadius", new Vector4(skyOuterRadius, skyOuterRadius * skyOuterRadius, skyInnerRadius, skyInnerRadius * skyInnerRadius));
            Shader.SetGlobalVector("_WeatherMakerSkyMie", new Vector4(1.5f * ((1.0f - mieG2) / (2.0f + mieG2)), 1.0f + mieG2, 2.0f + mieG, 0.0f));
            Shader.SetGlobalVector("_WeatherMakerSkyLightScattering", new Vector4(sunRed, sunGreen, sunBlue, sunIntensity));
            Shader.SetGlobalVector("_WeatherMakerSkyLightPIScattering", new Vector4(pi4Red, pi4Green, pi4Blue, pi4Intensity));
            Shader.SetGlobalVector("_WeatherMakerSkyScale", new Vector4(scaleFactor, scaleDepth, scaleOverScaleDepth, skyCameraHeight));
        }

        private void SetShaderLightParameters()
        {
            Material.SetFloat("_DayMultiplier", WeatherMakerScript.Instance.DayNightScript.DayMultiplier);
            Material.SetFloat("_DawnDuskMultiplier", WeatherMakerScript.Instance.DayNightScript.DawnDuskMultiplier);
            Material.SetFloat("_NightMultiplier", WeatherMakerScript.Instance.DayNightScript.NightMultiplier);
            Material.SetFloat("_NightSkyMultiplier", Mathf.Max(1.0f - Mathf.Min(1.0f, SkyAtmosphereThickness), WeatherMakerScript.Instance.DayNightScript.NightMultiplier));
            Material.SetFloat("_NightVisibilityThreshold", NightVisibilityThreshold);
            Material.SetFloat("_NightIntensity", NightIntensity);
            Material.DisableKeyword("ENABLE_NIGHT_TWINKLE");

            if (NightTwinkleRandomness > 0.0f || (NightTwinkleVariance > 0.0f && NightTwinkleSpeed > 0.0f))
            {
                Material.SetFloat("_NightTwinkleSpeed", NightTwinkleSpeed);
                Material.SetFloat("_NightTwinkleVariance", NightTwinkleVariance);
                Material.SetFloat("_NightTwinkleMinimum", NightTwinkleMinimum);
                Material.SetFloat("_NightTwinkleRandomness", NightTwinkleRandomness);
                Material.EnableKeyword("ENABLE_NIGHT_TWINKLE");
            }
        }

        private void RaycastForEclipse()
        {
            if (WeatherMakerScript.Instance.CurrentCamera == null)
            {
                return;
            }

            // disable allow eclipses everywhere by default
            float eclipsePower = 0.0f;
            foreach (WeatherMakerCelestialObject moon in WeatherMakerScript.Instance.Moons)
            {
                moon.Renderer.sharedMaterial.DisableKeyword("ENABLE_SUN_ECLIPSE");
                if (moon.Collider != null)
                {
                    moon.Collider.enabled = CheckForEclipse;
                }
            }

            if (CheckForEclipse)
            {
                float sunRadius = Mathf.Lerp(0.0f, 1000.0f, Mathf.Pow(WeatherMakerScript.Instance.Sun.Scale, 0.5f));
                Vector3 origin = WeatherMakerScript.Instance.CurrentCamera.transform.position - (WeatherMakerScript.Instance.Sun.Transform.forward * WeatherMakerScript.Instance.CurrentCamera.farClipPlane * 1.7f);
                int hitCount = Physics.SphereCastNonAlloc(origin, sunRadius, WeatherMakerScript.Instance.Sun.Transform.forward, eclipseHits, WeatherMakerScript.Instance.CurrentCamera.farClipPlane);
                for (int i = 0; i < hitCount; i++)
                {
                    foreach (WeatherMakerCelestialObject moon in WeatherMakerScript.Instance.Moons)
                    {
                        if (moon.Transform.gameObject == eclipseHits[i].collider.gameObject)
                        {
                            float dot = Mathf.Abs(Vector3.Dot(eclipseHits[i].normal, WeatherMakerScript.Instance.Sun.Transform.forward));
                            eclipsePower += Mathf.Pow(dot, 256.0f);
                            if (!moon.Renderer.sharedMaterial.IsKeywordEnabled("ENABLE_SUN_ECLIPSE"))
                            {
                                moon.Renderer.sharedMaterial.EnableKeyword("ENABLE_SUN_ECLIPSE");
                            }
                            //Debug.LogFormat("Eclipse raycast normal: {0}, dot: {1}, power: {2}", eclipseHits[i].normal, dot, eclipsePower);
                            break;
                        }
                    }
                }
            }

            if (eclipsePower == 0.0f)
            {
                WeatherMakerScript.Instance.DayNightScript.DirectionalLightIntensityMultipliers["WeatherMakerSkySphereScriptEclipse"] = 1.0f;
            }
            else
            {
                float eclipseLightReducer = 1.0f - Mathf.Clamp(eclipsePower, 0.0f, 1.0f);
                WeatherMakerScript.Instance.DayNightScript.DirectionalLightIntensityMultipliers["WeatherMakerSkySphereScriptEclipse"] = eclipseLightReducer;
                Material.SetFloat("_NightSkyMultiplier", Mathf.Max(1.0f - Mathf.Min(1.0f, SkyAtmosphereThickness), Mathf.Max(eclipsePower, WeatherMakerScript.Instance.DayNightScript.NightMultiplier)));
            }
        }

        private void SetSkySphereScalesAndPositions()
        {
            if (WeatherMakerScript.Instance.CurrentCamera == null)
            {
                return;
            }

            // adjust sky sphere position and scale
            float farPlane = WeatherMakerScript.Instance.CurrentCamera.farClipPlane * FarClipPlaneMultiplier * 0.9f;
            Vector3 anchor = WeatherMakerScript.Instance.CurrentCamera.transform.position;
            float yOffset = farPlane * YOffsetMultiplier;
            gameObject.transform.position = anchor + new Vector3(0.0f, yOffset, 0.0f);
            float scale = farPlane * ((WeatherMakerScript.Instance.CurrentCamera.farClipPlane - Mathf.Abs(yOffset)) / WeatherMakerScript.Instance.CurrentCamera.farClipPlane);
            float finalScale;
            gameObject.transform.localScale = new Vector3(scale, scale, scale);

            // move sun back near the far plane and scale appropriately
            Vector3 sunOffset = (WeatherMakerScript.Instance.Sun.Transform.forward * ((farPlane * 0.9f) - scale));
            WeatherMakerScript.Instance.Sun.Transform.position = anchor - sunOffset;
            WeatherMakerScript.Instance.Sun.Transform.localScale = new Vector3(scale, scale, scale);

            // move moons back near the far plane and scale appropriately
            foreach (WeatherMakerCelestialObject moon in WeatherMakerScript.Instance.Moons)
            {
                scale = farPlane * moon.Scale;
                finalScale = Mathf.Clamp(Mathf.Abs(moon.Transform.forward.y) * 3.0f, 0.8f, 1.0f);
                finalScale = scale / finalScale;
                Vector3 moonOffset = (moon.Transform.forward * (farPlane - finalScale));
                moon.Transform.position = anchor - moonOffset;
                moon.Transform.localScale = new Vector3(finalScale, finalScale, finalScale);
            }
        }

        private void UpdateSkySphere()
        {
            WeatherMakerScript.Instance.CurrentCamera = (Camera.current == null ? WeatherMakerScript.Instance.Camera : Camera.current);
            SetSkySphereScalesAndPositions();
            SetShaderSkyParameters();
            SetShaderLightParameters();
            RaycastForEclipse();
        }

        internal void PreCullCamera(Camera c)
        {

#if UNITY_EDITOR

            if (Application.isPlaying)
            {

#endif

                UpdateSkySphere();

#if UNITY_EDITOR

            }

#endif

        }

        protected override void OnWillRenderObject()
        {
            base.OnWillRenderObject();

#if UNITY_EDITOR

            if (!Application.isPlaying)
            {
                UpdateSkySphere();
            }

#endif

        }

        protected override void Start()
        {
            base.Start();
            UpdateSkySphere();
        }
    }
}