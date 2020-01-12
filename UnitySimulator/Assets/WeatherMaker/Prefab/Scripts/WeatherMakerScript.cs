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
    public enum WeatherMakerPrecipitationType
    {
        None = 0,
        Rain = 1,
        Snow = 2,
        Sleet = 3,
        Hail = 4,
        Custom = 127
    }

    public enum WeatherMakerCloudType
    {
        None = 0,
        Light = 1,
        Medium = 2,
        Heavy = 3,
        Storm = 4,
        HeavyBright = 5
    }

    public enum WeatherMakerOrbitType
    {
        /// <summary>
        /// Orbit as viewed from Earth
        /// </summary>
        FromEarth = 0,

        /// <summary>
        /// Orbit is controlled by script implementing IWeatherMakerCustomOrbit interface
        /// </summary>
        Custom
    }

    public enum BlurShaderType
    {
        None,
        GaussianBlur7,
        GaussianBlur17
    }

    /// <summary>
    /// Interface for custom orbits
    /// </summary>
    public interface IWeatherMakerCelestialObjectCustomOrbit
    {
        /// <summary>
        /// Calculate a custom orbit for a celestial object
        /// </summary>
        /// <param name="obj">Celestial object</param>
        /// <returns>Normalized (unit) vector pointing down from the sky to origin (0,0,0) for object position in sky</returns>
        Vector3 CalculatePosition(WeatherMakerCelestialObject obj);
    }

    [System.Serializable]
    public class WeatherMakerCelestialObject
    {
        [Tooltip("The transform that is orbiting")]
        public Transform Transform;

        [Tooltip("The directional light used to emit light from the object")]
        public Light Light;

        [Tooltip("The renderer for the object, must not be null")]
        public Renderer Renderer;

        [Tooltip("The collider for the oject, can be null")]
        public Collider Collider;

        [Tooltip("Hint to have the object render in fast mode. Useful for mobile, but not all shaders support it.")]
        public bool RenderHintFast;

        [Tooltip("Rotation about y axis - changes how the celestial body orbits over the scene")]
        public float RotateYDegrees;

        [Tooltip("The orbit type. Only from Earth orbit is currently supported.")]
        public WeatherMakerOrbitType OrbitType;

        [Tooltip("Script to use if OrbitType is custom, otherwise ignored.")]
        public MonoBehaviour OrbitTypeCustomScript;

        [Range(0.0f, 1.0f)]
        [Tooltip("The scale of the object. For the sun, this is shader specific. For moons, this is a percentage of camera far plane.")]
        public float Scale = 0.03f;

        [Tooltip("Light color")]
        public Color LightColor = Color.white;

        [Range(0.0f, 128.0f)]
        [Tooltip("Light power, controls how intense the light lights up the clouds, etc. near the object. Lower values reduce the radius and increase the intensity.")]
        public float LightPower = 8.0f;

        [Tooltip("The intensity of the light of the object at default (full) intensity")]
        [Range(0.0f, 3.0f)]
        public float LightBaseIntensity = 1.0f;

        [Tooltip("The shadow strength of the light of the object at default (full) shadow intensity")]
        [Range(0.0f, 1.0f)]
        public float LightBaseShadowStrength = 0.8f;

        [Range(0.0f, 3.0f)]
        [Tooltip("Light multiplier")]
        public float LightMultiplier = 1.0f;

        [Tooltip("Tint color of the object.")]
        public Color TintColor = Color.white;

        [Range(0.0f, 4.0f)]
        [Tooltip("Tint intensity")]
        public float TintIntensity = 1.0f;

        /// <summary>
        /// Whether the object is active
        /// </summary>
        public bool IsActive
        {
            get { return Transform.gameObject.activeInHierarchy && Scale > 0.0f; }
        }

        /// <summary>
        /// Whether the light for this object is active. A light that is not active is not on.
        /// </summary>
        public bool LightIsActive
        {
            get { return Light != null && Light.enabled && IsActive; }
        }

        /// <summary>
        /// Whether the light is on. An active light can have a light that is off.
        /// </summary>
        public bool LightIsOn
        {
            get { return LightIsActive && Light.intensity > 0.0f && LightMultiplier > 0.0001f && Light.color.r > 0.0001f && Light.color.g > 0.0001f && Light.color.b > 0.0001f; }
        }

        /// <summary>
        /// Gets the viewport positions of the object, for each eye. May be null if not set.
        /// </summary>
        public Vector3 ViewportPosition
        {
            get; internal set;
        }
    }

    [ExecuteInEditMode]
    public class WeatherMakerScript : MonoBehaviour
    {
        [Header("Configuration")]
        [Tooltip("Camera the weather should render around. Defaults to main camera.")]
        public Camera Camera;

        [Tooltip("Additional cameras. Do not add the primary camera (i.e. the Camera property) to this array, only add extra cameras. These cameras will get all weather effects including sky sphere, rain, fog ,etc.")]
        public System.Collections.Generic.List<Camera> Cameras;

        [Tooltip("Optional weather profile, used to set properties from a scriptable object.")]
        [SerializeField]
        private WeatherMakerProfileScript _WeatherProfile;

#if UNITY_EDITOR

        /// <summary>
        /// Allow changing the profile via inspector in editor only
        /// </summary>
        private WeatherMakerProfileScript lastProfile;

#endif

        public WeatherMakerProfileScript WeatherProfile
        {
            get { return _WeatherProfile; }
            set
            {
                if (_WeatherProfile != value)
                {
                    if (WeatherProfileChanged != null)
                    {
                        WeatherProfileChanged.Invoke(_WeatherProfile, value);
                    }
                    _WeatherProfile = value;
                    UpdateWeatherProfile();
                }
            }
        }

        /// <summary>
        /// Event that fires when the weather profile changes
        /// </summary>
        public event System.Action<WeatherMakerProfileScript, WeatherMakerProfileScript> WeatherProfileChanged;

        /// <summary>
        /// Executes when a weather manager starts a new transition. Parameters are index into array of WeatherManagers, index of transition group in Transitions property and random seed.
        /// </summary>
        public event System.Action<int, int, int> WeatherManagerTransitionStarted;

        internal void RaiseWeatherManagerTransitionStarted(int managerIndex, int transitionIndex, int randomSeed)
        {
            if (WeatherMakerScript.Instance.WeatherManagerTransitionStarted != null)
            {
                WeatherMakerScript.Instance.WeatherManagerTransitionStarted.Invoke(managerIndex, transitionIndex, randomSeed);
            }
        }

        [Tooltip("Configuration script. Should be deactivated for release builds.")]
        public WeatherMakerConfigurationScript ConfigurationScript;

        [Range(0.0f, 1.0f)]
        [Tooltip("Change the volume of all weather maker sounds.")]
        [SerializeField]
        [UnityEngine.Serialization.FormerlySerializedAs("VolumeModifier")]
        private float volumeModifier = 1.0f;

        internal float cachedVolumeModifier;

        /// <summary>
        /// Get or set the volume modifier. Note that setting the volume modifier only sets the 'WeatherMakerScript' volume modifier in VolumeModifierDictionary.
        /// There may be other additional volume modifiers that get applied when you get the value of VolumeModifier.
        /// </summary>
        public float VolumeModifier
        {
            get
            {
                float baseVolume = 1.0f;
                foreach (float val in VolumeModifierDictionary.Values)
                {
                    baseVolume *= val;
                }
                return baseVolume;
            }
            set
            {
                VolumeModifierDictionary["WeatherMakerScript"] = value;
            }
        }

        [Tooltip("Whether per pixel lighting is enabled - currently precipitation mist is the only material that support this.")]
        public bool EnablePerPixelLighting;

        [Range(-100.0f, 150.0f)]
        [Tooltip("Temperature, not used yet")]
        public float Temperature = 70.0f;

        [Range(0.0f, 1.0f)]
        [Tooltip("Humidity, not used yet")]
        public float Humidity = 0.1f;

        [Header("Celestial Objects")]
        [Tooltip("All suns in the scene - must have at least one. *Only one sun is currently supported*. All items must not be null.")]
        public WeatherMakerCelestialObject[] Suns;

        [Tooltip("(3D only) All moons in the scene. All items must not be null.")]
        public WeatherMakerCelestialObject[] Moons;

        private WeatherMakerPrecipitationType precipitation = WeatherMakerPrecipitationType.None;
        [Header("Precipitation")]
        [Tooltip("Current precipitation")]
        public WeatherMakerPrecipitationType Precipitation = WeatherMakerPrecipitationType.None;

        [Tooltip("Intensity of precipitation (0-1)")]
        [Range(0.0f, 1.0f)]
        public float PrecipitationIntensity;

        [Tooltip("How long in seconds to fully change from one precipitation type to another")]
        [Range(0.0f, 300.0f)]
        public float PrecipitationChangeDuration = 4.0f;

        [Tooltip("How long to delay before applying a change in precipitation intensity.")]
        [Range(0.0f, 300.0f)]
        public float PrecipitationChangeDelay = 0.0f;

        [Tooltip("The threshold change in intensity that will cause a cross-fade between precipitation changes. Intensity changes smaller than this value happen quickly.")]
        [Range(0.0f, 0.2f)]
        public float PrecipitationChangeThreshold = 0.1f;

        [Tooltip("Whether the precipitation collides with the world. This can be a performance problem on lower end devices. Please be careful.")]
        public bool PrecipitationCollisionEnabled;

        private WeatherMakerCloudType clouds = WeatherMakerCloudType.None;
        [Header("Clouds")]
        [Tooltip("Cloud type. In 2D, clouds are either none or any other option will be storm.")]
        public WeatherMakerCloudType Clouds = WeatherMakerCloudType.None;

        [Tooltip("How long in seconds to fully change from one cloud type to another, does not apply in 2D")]
        [Range(0.0f, 300.0f)]
        public float CloudChangeDuration = 4.0f;

        [Header("Dependencies")]
        [Tooltip("Rain script")]
        public WeatherMakerFallingParticleScript RainScript;

        [Tooltip("Snow script")]
        public WeatherMakerFallingParticleScript SnowScript;

        [Tooltip("Hail script")]
        public WeatherMakerFallingParticleScript HailScript;

        [Tooltip("Sleet script")]
        public WeatherMakerFallingParticleScript SleetScript;

        [Tooltip("Set a custom precipitation script for use with Precipitation = WeatherMakerPrecipitationType.Custom ")]
        public WeatherMakerFallingParticleScript CustomPrecipitationScript;

        [Tooltip("Wind script")]
        public WeatherMakerWindScript WindScript;

        [Tooltip("Day night script")]
        public WeatherMakerDayNightCycleScript DayNightScript;

        [Tooltip("(3D) Sky sphere script, null if none")]
        public WeatherMakerSkySphereScript SkySphereScript;

        [Tooltip("(2D) Sky plane script, null if none")]
        public WeatherMakerSkyPlaneScript SkyPlaneScript;

        [Tooltip("Fog script, null if none")]
        public WeatherMakerFullScreenFogScript FogScript;

        [Tooltip("Cloud script, null if none")]
        public WeatherMakerFullScreenCloudsScript CloudScript;

        [Tooltip("Lightning script (random bolts)")]
        public WeatherMakerThunderAndLightningScript LightningScript;

        [Tooltip("Lightning bolt script")]
        public WeatherMakerLightningBoltScript LightningBoltScript;

        [Tooltip("Light manager, 3D only.")]
        public WeatherMakerLightManagerScript LightManagerScript;

        [Tooltip("Command buffer manager script")]
        public WeatherMakerCommandBufferManagerScript CommandBufferManagerScript;

        [Tooltip("A list of all the weather managers. Only one object should be active at a time. Use the ActivateWeatherManager method to switch managers.")]
        public System.Collections.Generic.List<WeatherMakerWeatherManagerScript> WeatherManagers;

        [Header("Deprecated Components, may be removed in future versions.")]
        [Tooltip("Legacy cloud script for 2D, null if none.")]
        public WeatherMakerLegacyCloudScript2D LegacyCloudScript2D;

        /// <summary>
        /// Allows adding additional volume modifiers by key
        /// </summary>
        [NonSerialized]
        public readonly System.Collections.Generic.Dictionary<string, float> VolumeModifierDictionary = new System.Collections.Generic.Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// The current precipitation script - use Precipitation to change precipitation
        /// </summary>
        public WeatherMakerFallingParticleScript PrecipitationScript { get; private set; }

        /// <summary>
        /// Get / set the intensity of the wind. Simply a shortcut to WindScript.WindIntensity.
        /// </summary>
        public float WindIntensity { get { return WindScript.WindIntensity; } set { WindScript.WindIntensity = value; } }

        /// <summary>
        /// Gets the current time of day in seconds. 86400 seconds per day.
        /// </summary>
        public float TimeOfDay { get { return DayNightScript.TimeOfDay; } set { DayNightScript.TimeOfDay = value; } }

        /// <summary>
        /// Returns the first object in Suns
        /// </summary>
        public WeatherMakerCelestialObject Sun { get { return Suns[0]; } }

        /// <summary>
        /// Returns the first object in Moons
        /// </summary>
        public WeatherMakerCelestialObject Moon { get { return (Moons.Length == 0 ? null : Moons[0]); } }

        /// <summary>
        /// The current camera (used by some scripts in OnWillRenderObject, PreCull, etc.)
        /// </summary>
        internal Camera CurrentCamera { get; set; }

        /// <summary>
        /// Current camera position
        /// </summary>
        internal Vector3 CurrentCameraPosition { get; set; }

        /// <summary>
        /// Returns whether the Camera property is orthographic
        /// </summary>
        public bool CameraIsOrthographic { get { return Camera != null && Camera.orthographic; } }

        /// <summary>
        /// The planes of the current camera view frustum
        /// </summary>
        public Plane[] CurrentCameraFrustumPlanes { get; private set; }

        /// <summary>
        /// Current network script, will be an empty script if not networking
        /// </summary>
        public IWeatherMakerNetworkScript NetworkScript
        {
            get { return (_networkScript ?? (_networkScript = new WeatherMakerNullNetworkScript())); }
            set { _networkScript = value; }
        }
        private IWeatherMakerNetworkScript _networkScript;

        public static WeatherMakerScript Instance { get; private set; }

        private float lastPrecipitationIntensityChange = -1.0f;

        [SerializeField]
        [HideInInspector]
        internal bool multiPassStereoRenderingEnabled;

        /// <summary>
        /// Max number of moons supported. This should match the constant in WeatherMakerShader.cginc.
        /// </summary>
        public const int MaxMoonCount = 8;

        private readonly Vector4[] moonDirectionUpShaderBuffer = new Vector4[MaxMoonCount];
        private readonly Vector4[] moonDirectionDownShaderBuffer = new Vector4[MaxMoonCount];
        private readonly Vector4[] moonLightColorShaderBuffer = new Vector4[MaxMoonCount];
        private readonly Vector4[] moonLightPowerShaderBuffer = new Vector4[MaxMoonCount];
        private readonly Vector4[] moonTintColorShaderBuffer = new Vector4[MaxMoonCount];
        private readonly Vector4[] moonVar1ShaderBuffer = new Vector4[MaxMoonCount];
        private readonly System.Collections.Generic.List<System.Action> mainThreadActions = new System.Collections.Generic.List<Action>();

        /// <summary>
        /// Turn off all weather managers
        /// </summary>
        public void DeactivateWeatherManagers()
        {
            if (WeatherManagers != null)
            {
                foreach (WeatherMakerWeatherManagerScript manager in WeatherManagers)
                {
                    manager.gameObject.SetActive(false);
                }
            }
        }

        /// <summary>
        /// Queue an action to run on the main thread - this action should run as fast as possible
        /// </summary>
        /// <param name="action">Action to run</param>
        public void QueueOnMainThread(System.Action action)
        {
            lock (mainThreadActions)
            {
                mainThreadActions.Add(action);
            }
        }

        /// <summary>
        /// Activate the nth weather manager in the WeatherManagers list, deactivating the current weather manager and activating the selected weather manager
        /// </summary>
        /// <param name="index">Index of the new weather manager to activate</param>
        /// <returns>True if success, false if index out of bounds</returns>
        public bool ActivateWeatherManager(int index)
        {
            if (WeatherManagers != null && index < WeatherManagers.Count)
            {
                DeactivateWeatherManagers();
                WeatherManagers[index].gameObject.SetActive(true);
                return true;
            }
            return false;
        }
        
        public static bool AssertExists()
        {

#if UNITY_EDITOR

            if (WeatherMakerScript.Instance == null || WeatherMakerScript.Instance.DayNightScript == null)
            {
                Debug.LogError("WeatherMakerPrefab missing from scene (WeatherMakerScript and WeatherMakerDayNightCycleScript are required and must not be deleted).");
                return false;
            }

#endif

            return true;
        }

        private void TweenScript(WeatherMakerFallingParticleScript script, float end)
        {
            if (PrecipitationChangeDuration < 0.1f)
            {
                script.Intensity = end;
                return;
            }

            float duration = (Mathf.Abs(script.Intensity - end) < PrecipitationChangeThreshold ? 0.0f : PrecipitationChangeDuration);
            FloatTween tween = TweenFactory.Tween("WeatherMakerPrecipitationChange_" + script.gameObject.GetInstanceID(), script.Intensity, end, duration, TweenScaleFunctions.Linear, (t) =>
            {
                // Debug.LogFormat("Tween key: {0}, value: {1}, prog: {2}", t.Key, t.CurrentValue, t.CurrentProgress);
                script.Intensity = t.CurrentValue;
            });
            tween.Delay = PrecipitationChangeDelay;
        }

        private void ChangePrecipitation(WeatherMakerFallingParticleScript newPrecipitation)
        {
            if (newPrecipitation != PrecipitationScript && PrecipitationScript != null)
            {
                TweenScript(PrecipitationScript, 0.0f);
                lastPrecipitationIntensityChange = -1.0f;
            }
            PrecipitationScript = newPrecipitation;
        }

        private void UpdateCollision()
        {
            if (RainScript != null)
            {
                RainScript.CollisionEnabled = PrecipitationCollisionEnabled;
            }
            if (SnowScript != null)
            {
                SnowScript.CollisionEnabled = PrecipitationCollisionEnabled;
            }
            if (HailScript != null)
            {
                HailScript.CollisionEnabled = PrecipitationCollisionEnabled;
            }
            if (SleetScript != null)
            {
                SleetScript.CollisionEnabled = PrecipitationCollisionEnabled;
            }
            if (CustomPrecipitationScript != null)
            {
                CustomPrecipitationScript.CollisionEnabled = PrecipitationCollisionEnabled;
            }
        }

        private void SetVolumeModifier(WeatherMakerFallingParticleScript script, float volumeModifier)
        {
            if (script != null)
            {
                script.SetVolumeModifier(volumeModifier);
            }
        }

        private void UpdateSoundsVolumes()
        {
            cachedVolumeModifier = VolumeModifier;
            SetVolumeModifier(RainScript, cachedVolumeModifier);
            SetVolumeModifier(SnowScript, cachedVolumeModifier);
            SetVolumeModifier(HailScript, cachedVolumeModifier);
            SetVolumeModifier(SleetScript, cachedVolumeModifier);
            SetVolumeModifier(CustomPrecipitationScript, cachedVolumeModifier);
            if (WindScript != null && WindScript.AudioSourceWind != null)
            {
                WindScript.AudioSourceWind.VolumeModifier = cachedVolumeModifier;
            }
            if (LightningScript != null)
            {
                LightningScript.VolumeModifier = cachedVolumeModifier;
            }
        }

        private void SetEnablePerPixelLighting()
        {

#if UNITY_EDITOR

            if (!Application.isPlaying)
            {
                return;
            }

#endif

            if (EnablePerPixelLighting && SystemInfo.graphicsShaderLevel >= 30)
            {
                if (!Shader.IsKeywordEnabled("WEATHER_MAKER_PER_PIXEL_LIGHTING"))
                {
                    Shader.EnableKeyword("WEATHER_MAKER_PER_PIXEL_LIGHTING");
                }
            }
            else if (Shader.IsKeywordEnabled("WEATHER_MAKER_PER_PIXEL_LIGHTING"))
            {
                Shader.DisableKeyword("WEATHER_MAKER_PER_PIXEL_LIGHTING");
            }
        }

        private void SetGlobalShaders()
        {
            if (Sun != null)
            {
                // Sun
                Vector3 sunForward = Sun.Transform.forward;
                Vector3 sunForward2D = Quaternion.AngleAxis(-90.0f, Vector3.right) * Sun.Transform.forward;
                Shader.SetGlobalVector("_WeatherMakerSunDirectionUp", -sunForward);
                Shader.SetGlobalVector("_WeatherMakerSunDirectionUp2D", -sunForward2D);
                Shader.SetGlobalVector("_WeatherMakerSunDirectionDown", Sun.Transform.forward);
                Shader.SetGlobalVector("_WeatherMakerSunDirectionDown2D", sunForward2D);
                Shader.SetGlobalVector("_WeatherMakerSunPositionNormalized", Sun.Transform.position.normalized);
                Shader.SetGlobalVector("_WeatherMakerSunPositionWorldSpace", Sun.Transform.position);
                Vector4 sunColor = new Vector4(Sun.Light.color.r, Sun.Light.color.g, Sun.Light.color.b, Sun.Light.intensity);
                Shader.SetGlobalVector("_WeatherMakerSunColor", sunColor);
                sunColor = new Vector4(Sun.TintColor.r, Sun.TintColor.g, Sun.TintColor.b, Sun.TintColor.a * Sun.TintIntensity * Sun.Light.intensity);
                Shader.SetGlobalVector("_WeatherMakerSunTintColor", sunColor);
                float sunHorizonScaleMultiplier = Mathf.Clamp(Mathf.Abs(Sun.Transform.forward.y) * 3.0f, 0.5f, 1.0f);
                sunHorizonScaleMultiplier = Mathf.Min(1.0f, Sun.Scale / sunHorizonScaleMultiplier);
                Shader.SetGlobalVector("_WeatherMakerSunLightPower", new Vector4(Sun.LightPower, Sun.LightMultiplier, Sun.Light.shadowStrength, 1.0f - Sun.Light.shadowStrength));
                Shader.SetGlobalVector("_WeatherMakerSunVar1", new Vector4(sunHorizonScaleMultiplier, Mathf.Pow(Sun.Light.intensity, 0.5f), Mathf.Pow(Sun.Light.intensity, 0.75f), Sun.Light.intensity * Sun.Light.intensity));

                if (Sun.Renderer != null)
                {
                    if (Sun.RenderHintFast)
                    {
                        Sun.Renderer.sharedMaterial.EnableKeyword("RENDER_HINT_FAST");
                    }
                    else
                    {
                        Sun.Renderer.sharedMaterial.DisableKeyword("RENDER_HINT_FAST");
                    }
                }
            }

            // Moons
            Shader.SetGlobalInt("_WeatherMakerMoonCount", Moons.Length);
            for (int i = 0; i < Moons.Length; i++)
            {
                WeatherMakerCelestialObject moon = Moons[i];
                moon.Renderer.sharedMaterial.SetFloat("_MoonIndex", i);
                moonDirectionUpShaderBuffer[i] = -moon.Transform.forward;
                moonDirectionDownShaderBuffer[i] = moon.Transform.forward;
                moonLightColorShaderBuffer[i] = (moon.LightIsOn ? new Vector4(moon.Light.color.r, moon.Light.color.g, moon.Light.color.b, moon.Light.intensity) : Vector4.zero);
                moonLightPowerShaderBuffer[i] = new Vector4(moon.LightPower, moon.LightMultiplier, moon.Light.shadowStrength, 1.0f - moon.Light.shadowStrength);
                moonTintColorShaderBuffer[i] = new Vector4(moon.TintColor.r, moon.TintColor.g, moon.TintColor.b, moon.TintColor.a * moon.TintIntensity);
                moonVar1ShaderBuffer[i] = new Vector4(moon.Scale, 0.0f, 0.0f, 0.0f);
            }

            Shader.SetGlobalVectorArray(WeatherMakerShaderIds.ArrayWeatherMakerMoonDirectionUp, moonDirectionUpShaderBuffer);
            Shader.SetGlobalVectorArray(WeatherMakerShaderIds.ArrayWeatherMakerMoonDirectionDown, moonDirectionDownShaderBuffer);
            Shader.SetGlobalVectorArray(WeatherMakerShaderIds.ArrayWeatherMakerMoonLightColor, moonLightColorShaderBuffer);
            Shader.SetGlobalVectorArray(WeatherMakerShaderIds.ArrayWeatherMakerMoonLightPower, moonLightPowerShaderBuffer);
            Shader.SetGlobalVectorArray(WeatherMakerShaderIds.ArrayWeatherMakerMoonTintColor, moonTintColorShaderBuffer);
            Shader.SetGlobalVectorArray(WeatherMakerShaderIds.ArrayWeatherMakerMoonVar1, moonVar1ShaderBuffer);

            float t = Time.timeSinceLevelLoad;
            Shader.SetGlobalVector("_WeatherMakerTime", new Vector4(t * 0.05f, t, (float)System.Math.Truncate(t * 0.05f), (float)System.Math.Truncate(t)));
            Shader.SetGlobalVector("_WeatherMakerTimeSin", new Vector4(Mathf.Sin(t * 0.05f), Mathf.Sin(t), Mathf.Sin(t * 2.0f), Mathf.Sin(t * 3.0f)));
        }

        private void SetShaderSunViewportPosition(Camera camera)
        {
            if (Sun == null)
            {
                return;
            }
            Sun.ViewportPosition = camera.WorldToViewportPoint(Sun.Transform.position);
            Shader.SetGlobalVector("_WeatherMakerSunViewportPosition", Sun.ViewportPosition);
        }

        private void CameraPreCull(Camera camera)
        {
            if ((camera.depthTextureMode & DepthTextureMode.Depth) == DepthTextureMode.None)
            {
                camera.depthTextureMode |= DepthTextureMode.Depth;
            }
            CurrentCamera = camera;
            CurrentCameraPosition = camera.transform.position;
            SetShaderSunViewportPosition(camera);
            CurrentCameraFrustumPlanes = GeometryUtility.CalculateFrustumPlanes(camera);
            if (RainScript != null && RainScript.isActiveAndEnabled)
            {
                RainScript.PreCullCamera(camera);
            }
            if (SnowScript != null && SnowScript.isActiveAndEnabled)
            {
                SnowScript.PreCullCamera(camera);
            }
            if (HailScript != null && HailScript.isActiveAndEnabled)
            {
                HailScript.PreCullCamera(camera);
            }
            if (SleetScript != null && SleetScript.isActiveAndEnabled)
            {
                SleetScript.PreCullCamera(camera);
            }
            if (CustomPrecipitationScript != null && CustomPrecipitationScript.isActiveAndEnabled)
            {
                CustomPrecipitationScript.PreCullCamera(camera);
            }
            if (SkySphereScript != null && SkySphereScript.isActiveAndEnabled)
            {
                SkySphereScript.PreCullCamera(camera);
            }
            if (CloudScript != null)
            {
                CloudScript.PreCullCamera(camera);
            }
            if (SkyPlaneScript != null && SkyPlaneScript.isActiveAndEnabled)
            {
                SkyPlaneScript.PreCullCamera(camera);
            }
            if (FogScript != null)
            {
                FogScript.PreCullCamera(camera);
            }
            if (LightManagerScript != null && LightManagerScript.isActiveAndEnabled)
            {
                LightManagerScript.PrepareToRenderCamera(camera);
            }
            if (CommandBufferManagerScript != null && CommandBufferManagerScript.isActiveAndEnabled)
            {
                CommandBufferManagerScript.PreCullCamera(camera);
            }
        }

        private void UpdateShaders()
        {
            SetEnablePerPixelLighting();
            SetGlobalShaders();
        }

        private void UpdateCameras()
        {
            Camera = (Camera == null ? Camera.main : Camera);

#if UNITY_EDITOR

            if (!Application.isPlaying)
            {
                return;
            }

#endif

            foreach (Camera c in Camera.allCameras)
            {
                if (c.GetComponent<WeatherMakerCameraPreCullScript>() == null && (c == Camera || Cameras.Contains(c)))
                {
                    WeatherMakerCameraPreCullScript script = c.gameObject.AddComponent<WeatherMakerCameraPreCullScript>();
                    script.hideFlags = HideFlags.HideAndDontSave | HideFlags.HideInInspector;
                    script.PreCull += CameraPreCull;
                }
            }
        }

        private void UpdateClouds()
        {
            if (clouds == Clouds)
            {
                return;
            }
            clouds = Clouds;
            if (CameraIsOrthographic)
            {
                if (LegacyCloudScript2D != null)
                {
                    if (clouds == WeatherMakerCloudType.None)
                    {
                        LegacyCloudScript2D.RemoveClouds();
                    }
                    else
                    {
                        LegacyCloudScript2D.CreateClouds();
                    }
                }
            }
            else if (SkySphereScript == null)
            {
                return;
            }
            else if (clouds == WeatherMakerCloudType.None)
            {
                CloudScript.HideCloudsAnimated(CloudChangeDuration);
            }
            else if (clouds == WeatherMakerCloudType.Light)
            {
                CloudScript.ShowCloudsAnimated(CloudChangeDuration, 0.17f, 0.0f, -1.0f, -1.0f, 0.05f, (_WeatherProfile == null || _WeatherProfile.Disabled ? (Color?)null : _WeatherProfile.CloudColor));
            }
            else if (clouds == WeatherMakerCloudType.Medium)
            {
                CloudScript.ShowCloudsAnimated(CloudChangeDuration, 0.34f, 0.0f, -1.0f, -1.0f, 0.03f, (_WeatherProfile == null || _WeatherProfile.Disabled ? (Color?)null : _WeatherProfile.CloudColor));
            }
            else if (clouds == WeatherMakerCloudType.Heavy)
            {
                CloudScript.ShowCloudsAnimated(CloudChangeDuration, 0.75f, 0.2f, -1.0f, -1.0f, 0.011f, (_WeatherProfile == null || _WeatherProfile.Disabled ? (Color?)null : _WeatherProfile.CloudColor));
            }
            else if (clouds == WeatherMakerCloudType.HeavyBright)
            {
                CloudScript.ShowCloudsAnimated(CloudChangeDuration, 0.68f, 0.0f, -1.0f, -1.0f, 0.0125f, (_WeatherProfile == null || _WeatherProfile.Disabled ? (Color?)null : _WeatherProfile.CloudColor));
            }
            else
            {
                // storm
                CloudScript.ShowCloudsAnimated(CloudChangeDuration, 1.0f, 0.8f, -1.0f, -1.0f, 0.01f, (_WeatherProfile == null || _WeatherProfile.Disabled ? (Color?)null : _WeatherProfile.CloudColor));
            }
        }

        private void CheckForPrecipitationChange()
        {
            if (Precipitation != precipitation)
            {
                precipitation = Precipitation;
                switch (precipitation)
                {
                    default:
                        ChangePrecipitation(null);
                        break;

                    case WeatherMakerPrecipitationType.Rain:
                        ChangePrecipitation(RainScript);
                        break;

                    case WeatherMakerPrecipitationType.Snow:
                        ChangePrecipitation(SnowScript);
                        break;

                    case WeatherMakerPrecipitationType.Hail:
                        ChangePrecipitation(HailScript);
                        break;

                    case WeatherMakerPrecipitationType.Sleet:
                        ChangePrecipitation(SleetScript);
                        break;

                    case WeatherMakerPrecipitationType.Custom:
                        ChangePrecipitation(CustomPrecipitationScript);
                        break;
                }
            }

            if (PrecipitationScript != null && PrecipitationIntensity != lastPrecipitationIntensityChange)
            {
                lastPrecipitationIntensityChange = PrecipitationIntensity;
                TweenScript(PrecipitationScript, PrecipitationIntensity);
            }
        }

        private void SetupReferences()
        {
            Instance = this;
            UpdateCollision();
            WeatherMakerShaderIds.Initialize();
            VolumeModifier = volumeModifier;
            Camera = (Camera == null ? Camera.main : Camera);
        }

        private void UpdateMainThreadActions()
        {
            lock (mainThreadActions)
            {
                foreach (System.Action action in mainThreadActions)
                {
                    action();
                }
                mainThreadActions.Clear();
            }
        }

        private void UpdateWeatherProfile()
        {
            if (_WeatherProfile == null || _WeatherProfile.Disabled)
            {
                return;
            }
            float transitionDuration = _WeatherProfile.TransitionDuration.Random();
            CloudChangeDuration = transitionDuration;
            Clouds = _WeatherProfile.CloudType;
            if (CloudScript != null)
            {
                Vector2 newCloudVelocity = UnityEngine.Random.insideUnitCircle * _WeatherProfile.CloudSpeed.Random();
                TweenFactory.Tween("WeatherMakerScriptProfileChangeCloudVelocity", CloudScript.CloudNoiseVelocity, newCloudVelocity, transitionDuration, TweenScaleFunctions.Linear, (progress) =>
                {
                    CloudScript.CloudNoiseVelocity = progress.CurrentValue;
                });
            }
            Precipitation = _WeatherProfile.Precipitation;
            PrecipitationIntensity = _WeatherProfile.PrecipitationIntensity;
            PrecipitationChangeDuration = transitionDuration * 0.5f;
            if (Clouds == WeatherMakerCloudType.None)
            {                
                PrecipitationChangeDelay = 0.0f;
            }
            else
            {
                PrecipitationChangeDelay = transitionDuration * 0.5f;
            }
            if (FogScript != null)
            {
                FogScript.FogMode = WeatherMakerFogMode.Linear;
                FogScript.TransitionFogDensity(FogScript.FogDensity, _WeatherProfile.FogDensity, CloudChangeDuration);
                FogScript.FogHeight = _WeatherProfile.FogHeight;
            }
            if (WindScript != null)
            {
                WindScript.AnimateWindIntensity(_WeatherProfile.WindIntensity, transitionDuration);
                WindScript.WindMaximumChangeRotation = _WeatherProfile.WindMaximumChangeRotation;
                WindScript.WindMainMultiplier = _WeatherProfile.WindMainMultiplier;
            }
            if (LightningScript != null)
            {
                FloatTween t = TweenFactory.Tween("WeatherMakerScriptProfileChangeLightning", 0.0f, 0.0f, 0.01f, TweenScaleFunctions.Linear, null, (completion) =>
                {
                    LightningScript.LightningIntenseProbability = _WeatherProfile.LightningIntenseProbability;
                    LightningScript.LightningIntervalTimeRange = _WeatherProfile.LightningIntervalTimeRange;
                    LightningScript.LightningForcedVisibilityProbability = _WeatherProfile.LightningForcedVisibilityProbability;
                    LightningScript.GroundLightningChance = _WeatherProfile.LightningGroundChance;
                    LightningScript.CloudLightningChance = _WeatherProfile.LightningCloudChance;
                    LightningScript.EnableLightning = _WeatherProfile.LightningEnabled;
                });
                t.Delay = PrecipitationChangeDelay;
            }
            GameObject player = GameObject.FindGameObjectWithTag("Player");
            if (player != null)
            {
                WeatherMakerSoundZoneScript soundZone = player.GetComponent<WeatherMakerSoundZoneScript>();
                if (soundZone != null)
                {
                    soundZone.Sounds.Clear();
                    soundZone.Sounds.AddRange(_WeatherProfile.Sounds);
                }
            }
        }

#if UNITY_EDITOR

        [UnityEditor.Callbacks.DidReloadScripts] 
        private static void OnScriptsReloaded()
        {
            Instance = GameObject.FindObjectOfType<WeatherMakerScript>();
        }

#endif

        private void Awake()
        {

#if UNITY_EDITOR

            string currBuildSettings = UnityEditor.PlayerSettings.GetScriptingDefineSymbolsForGroup(UnityEditor.EditorUserBuildSettings.selectedBuildTargetGroup);
            if (!currBuildSettings.Contains("WEATHER_MAKER_PRESENT"))
            {
                UnityEditor.PlayerSettings.SetScriptingDefineSymbolsForGroup(UnityEditor.EditorUserBuildSettings.selectedBuildTargetGroup, currBuildSettings + ";WEATHER_MAKER_PRESENT");
            }

#endif

            SetupReferences();
        }

        private void Start()
        {

#if UNITY_EDITOR

            multiPassStereoRenderingEnabled = (UnityEngine.XR.XRSettings.enabled && UnityEditor.PlayerSettings.stereoRenderingPath == UnityEditor.StereoRenderingPath.MultiPass);
            if (Application.isPlaying)
            {

#endif

                if (WeatherMakerLightManagerScript.Instance != null)
                {
                    // wire up lightning bolt lights to the light manager
                    LightningBoltScript.LightAddedCallback = LightAdded;
                    LightningBoltScript.LightRemovedCallback = LightRemoved;
                }

#if UNITY_EDITOR

            }

#endif

            UpdateCameras();

#if !UNITY_EDITOR

            // do the weather profile update for the initial weather profile
            UpdateWeatherProfile();

#endif

        }

        private void Update()
        {
            UpdateMainThreadActions();

#if UNITY_EDITOR

            if (transform.position != Vector3.zero || transform.localScale != Vector3.one || transform.rotation != Quaternion.identity)
            {
                Debug.LogError("For correct rendering, weather maker prefab should have position and rotation of 0, and scale of 1.");
            }

            if (_WeatherProfile != lastProfile)
            {
                if (WeatherProfileChanged != null)
                {
                    WeatherProfileChanged.Invoke(lastProfile, _WeatherProfile);
                }
                lastProfile = _WeatherProfile;
                UpdateWeatherProfile();
            }

            if (Application.isPlaying)
            {

#endif

                CheckForPrecipitationChange();
                UpdateCollision();
                UpdateClouds();

#if UNITY_EDITOR

            }

#endif

            UpdateShaders();
            UpdateCameras();
        }

        private void LateUpdate()
        {
            UpdateSoundsVolumes();
        }

        private void OnDestroy()
        {
            foreach (Camera c in Camera.allCameras)
            {
                WeatherMakerCameraPreCullScript script = c.GetComponent<WeatherMakerCameraPreCullScript>();
                if (script != null)
                {
                    script.PreCull -= CameraPreCull;
                }
            }

#if UNITY_EDITOR

            Instance = GameObject.FindObjectOfType<WeatherMakerScript>();

#endif

        }

        private void LightAdded(Light l)
        {
            WeatherMakerLightManagerScript.Instance.AddLight(l);
        }

        private void LightRemoved(Light l)
        {
            WeatherMakerLightManagerScript.Instance.RemoveLight(l);
        }
    }

    public class WeatherMakerCameraPreCullScript : MonoBehaviour
    {
        private Camera Camera;

        private void Start()
        {
            Camera = GetComponent<Camera>();
        }

        private void OnPreCull()
        {
            if (PreCull != null)
            {
                PreCull.Invoke(Camera);
            }
        }

        private void OnPreRender()
        {

        }

        public event System.Action<Camera> PreCull;
    }
}
