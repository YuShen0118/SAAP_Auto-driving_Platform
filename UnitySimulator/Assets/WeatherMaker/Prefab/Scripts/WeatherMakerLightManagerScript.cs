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

using System;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    /// <summary>
    /// Manages lights in world space for use in shaders - you do not need to add the directional light to the Lights list, it is done automatically
    /// </summary>
    public class WeatherMakerLightManagerScript : MonoBehaviour
    {
        [Tooltip("Whether to find all lights in the scene automatically if no Lights were added programatically. If false, you must manually add / remove lights using the Lights property. " +
            "To ensure correct behavior, do not change in script, set it once in the inspector and leave it. If this is true, AddLight and RemoveLight do nothing.")]
        public bool AutoFindLights = false;

        [Tooltip("A list of lights to automatically add to the light manager. Only used if AutoFindLights is false.")]
        public List<Light> AutoAddLights;

        [Tooltip("Spot light quadratic attenuation - default is 25")]
        [Range(0.0f, 1000.0f)]
        public float SpotLightQuadraticAttenuation = 25.0f;

        [Tooltip("Point light quadratic attenuation - default is 1000")]
        [Range(0.0f, 1000.0f)]
        public float PointLightQuadraticAttenuation = 1000.0f;

        [Tooltip("How intense is the scatter of directional light in the fog.")]
        [Range(0.0f, 100.0f)]
        public float FogDirectionalLightScatterIntensity = 8.0f;

        [Tooltip("How quickly fog point lights falloff from the center radius. High values fall-off more.")]
        [Range(0.0f, 4.0f)]
        public float FogSpotLightRadiusFalloff = 1.2f;

        [Tooltip("How much light intensity increases the light in a fog light. Higher values increase the brightness.")]
        [Range(1.0f, 4.0f)]
        public float FogLightIntensityPower = 2.0f;

        [Tooltip("How much the sun reduces fog lights. As sun intensity approaches 1, fog light intensity is reduced by this value.")]
        [Range(0.0f, 1.0f)]
        public float FogLightSunIntensityReducer = 0.8f;

        [Tooltip("Noise texture for fog and other 3D effects.")]
        public Texture3D NoiseTexture3D;

        /// <summary>
        /// Null fog zones - this is handled automatically as null fog zone prefabs are added
        /// </summary>
        public List<Bounds> FogNullZones { get; private set; }
        private readonly Vector4[] nullFogZoneArrayMin = new Vector4[MaximumNullFogZones];
        private readonly Vector4[] nullFogZoneArrayMax = new Vector4[MaximumNullFogZones];

        /// <summary>
        /// Global shared copy of NoiseTexture3D
        /// </summary>
        public static Texture3D NoiseTexture3DInstance { get; private set; }

        /// <summary>
        /// Max number of null fog zones - the n closest will be sent to shaders
        /// </summary>
        public const int MaximumNullFogZones = 8;

        /// <summary>
        /// Maximum number of lights to send to the Weather Maker shaders - reduce if you are having performance problems
        /// This should match the constant 'MAX_LIGHT_COUNT' in WeatherMakerShader.cginc
        /// </summary>
        public const int MaximumLightCount = 16;

        // all lights
        private Vector4[] lightPositions = new Vector4[MaximumLightCount];
        private Vector4[] lightSpotDirections = new Vector4[MaximumLightCount];
        private Vector4[] lightSpotEnds = new Vector4[MaximumLightCount];
        private Vector4[] lightColors = new Vector4[MaximumLightCount];
        private Vector4[] lightAtten = new Vector4[MaximumLightCount];

        // dir lights
        private Vector4[] lightPositionsDir = new Vector4[MaximumLightCount];
        private Vector4[] lightColorsDir = new Vector4[MaximumLightCount];
        private Vector4[] lightAttenDir = new Vector4[MaximumLightCount];

        // point lights
        private Vector4[] lightPositionsPoint = new Vector4[MaximumLightCount];
        private Vector4[] lightColorsPoint = new Vector4[MaximumLightCount];
        private Vector4[] lightAttenPoint = new Vector4[MaximumLightCount];

        // spot lights
        private Vector4[] lightPositionsSpot = new Vector4[MaximumLightCount];
        private Vector4[] lightSpotDirectionsSpot = new Vector4[MaximumLightCount];
        private Vector4[] lightSpotEndsSpot = new Vector4[MaximumLightCount];
        private Vector4[] lightColorsSpot = new Vector4[MaximumLightCount];
        private Vector4[] lightAttenSpot = new Vector4[MaximumLightCount];

        // unused, but needed for GetLightProperties for non-spot lights
        private Vector4 tmpDir = Vector3.zero;
        private Vector4 tmpEnd = Vector3.zero;

        /// <summary>
        /// A list of all the lights, sorted by importance of light
        /// </summary>
        private readonly List<Light> lights = new List<Light>();

        private bool IsPointLightVisibleInCamera(Light light, Camera camera)
        {
            float range = light.range + 1.0f;
            return GeometryUtility.TestPlanesAABB(WeatherMakerScript.Instance.CurrentCameraFrustumPlanes, new Bounds { center = light.transform.position, extents = new Vector3(range, range, range) });
        }

        private bool IsSpotLightVisibleInCamera(Light light, Camera camera)
        {
           float range = light.range + 1.0f;
           return GeometryUtility.TestPlanesAABB(WeatherMakerScript.Instance.CurrentCameraFrustumPlanes, new Bounds { center = light.transform.position + (light.transform.forward * range * 0.5f), extents = new Vector3(range, range, range) });
        }

        private bool GetLightProperties(Light light, Camera camera, ref Vector4 pos, ref Vector4 atten, ref Vector4 color, ref Vector4 dir, ref Vector4 end)
        {
            if (light == null || WeatherMakerScript.Instance.CurrentCameraFrustumPlanes == null || !light.enabled || light.color.a <= 0.001f || light.intensity <= 0.001f || light.range <= 0.001f)
            {
                return false;
            }

            switch (light.type)
            {
                case LightType.Directional:
                {
                    pos = -light.transform.forward;
                    pos.w = 0;
                    dir = Vector4.zero;
                    end = Vector4.zero;
                    atten = new Vector4(-1.0f, 1.0f, 0.0f, 0.0f);
                    color = new Vector4(light.color.r, light.color.g, light.color.b, light.intensity);
                    return true;
                }

                case LightType.Spot:
                {
                    if (!IsSpotLightVisibleInCamera(light, camera))
                    {
                        return false;
                    }

                    float radius = light.range * Mathf.Tan(0.5f * light.spotAngle * Mathf.Deg2Rad);
                    end = light.transform.position + (light.transform.forward * light.range); // center of cone base
                    float rangeSquared = Mathf.Sqrt((radius * radius) + (light.range * light.range));
                    end.w = rangeSquared * rangeSquared; // slant length squared
                    rangeSquared = light.range * light.range;
                    atten = new Vector4(Mathf.Cos(light.spotAngle * 0.5f * Mathf.Deg2Rad), 1.0f / Mathf.Cos(light.spotAngle * 0.25f * Mathf.Deg2Rad), SpotLightQuadraticAttenuation / rangeSquared, 1.0f / rangeSquared);
                    color = new Vector4(light.color.r, light.color.g, light.color.b, Mathf.Pow(light.intensity, FogLightIntensityPower));
                    pos = light.transform.position; // apex
                    pos.w = Mathf.Pow(light.spotAngle * Mathf.Deg2Rad / Mathf.PI, 0.5f); // falloff resistor, thinner angles do not fall off at edges
                    dir = light.transform.forward; // direction cone is facing from apex
                    dir.w = radius * radius; // radius at base squared
                    return true;
                }

                case LightType.Point:
                {
                    if (!IsPointLightVisibleInCamera(light, camera))
                    {
                        return false;
                    }

                    float rangeSquared = light.range * light.range;
                    pos = light.transform.position;
                    pos.w = rangeSquared;
                    dir = Vector4.zero;
                    end = Vector4.zero;
                    atten = new Vector4(-1.0f, 1.0f, PointLightQuadraticAttenuation / rangeSquared, 1.0f / rangeSquared);
                    color = new Vector4(light.color.r, light.color.g, light.color.b, Mathf.Pow(light.intensity, FogLightIntensityPower));
                    return true;
                }

                default:
                    return false;
            }
        }

        private bool SetLightAtIndex(Light light, Camera c, ref int lightIndex, ref int? nonDirLightIndex)
        {
            if (!GetLightProperties(light, c, ref lightPositions[lightIndex], ref lightAtten[lightIndex], ref lightColors[lightIndex], ref lightSpotDirections[lightIndex], ref lightSpotEnds[lightIndex]))
            {
                return false;
            }
            else if (light.type != LightType.Directional && nonDirLightIndex == null)
            {
                nonDirLightIndex = lightIndex;
            }
            lightIndex++;
            return true;
        }

        private int LightSorter(Light light1, Light light2)
        {
            // sort disabled or invisible lights to the back
            if (!light1.enabled || light1.intensity == 0.0f || light1.color.a == 0.0f)
            {
                return 1;
            }
            else if (!light2.enabled || light2.intensity == 0.0f || light2.color.a == 0.0f)
            {
                return -1;
            }
            // directional lights always come first and have highest priority
            else if (light1.type == LightType.Directional && light2.type == LightType.Directional)
            {
                return light1.intensity.CompareTo(light2.intensity);
            }
            else if (light1.type == LightType.Directional)
            {
                return -1;
            }
            else if (light2.type == LightType.Directional)
            {
                return 1;
            }

            // create total sum of distance, intensity and range to use as sort
            float mag1 = (Vector3.Distance(light1.transform.position, WeatherMakerScript.Instance.CurrentCameraPosition) - light1.range) * light1.intensity;
            float mag2 = (Vector3.Distance(light2.transform.position, WeatherMakerScript.Instance.CurrentCameraPosition) - light2.range) * light2.intensity;
            return mag1.CompareTo(mag2);
        }

        private int NullFogZoneSorter(Bounds b1, Bounds b2)
        {
            // sort by distance from camera
            float d1 = Vector3.SqrMagnitude(b1.center - WeatherMakerScript.Instance.CurrentCameraPosition);
            float d2 = Vector3.SqrMagnitude(b2.center - WeatherMakerScript.Instance.CurrentCameraPosition);
            return d1.CompareTo(d2);
        }

        private void SetLightsByTypeToShader(Camera camera)
        {
            int dirLightCount = 0;
            int pointLightCount = 0;
            int spotLightCount = 0;

            foreach (Light light in lights)
            {
                switch (light.type)
                {
                    case LightType.Directional:
                        if (dirLightCount < MaximumLightCount && GetLightProperties(light, camera, ref lightPositionsDir[dirLightCount], ref lightAttenDir[dirLightCount], ref lightColorsDir[dirLightCount], ref tmpDir, ref tmpEnd))
                        {
                            dirLightCount++;
                        }
                        break;

                    case LightType.Point:
                        if (pointLightCount < MaximumLightCount && GetLightProperties(light, camera, ref lightPositionsPoint[pointLightCount], ref lightAttenPoint[pointLightCount], ref lightColorsPoint[pointLightCount], ref tmpDir, ref tmpEnd))
                        {
                            pointLightCount++;
                        }
                        break;

                    case LightType.Spot:
                        if (spotLightCount < MaximumLightCount && GetLightProperties(light, camera, ref lightPositionsSpot[spotLightCount], ref lightAttenSpot[spotLightCount], ref lightColorsSpot[spotLightCount], ref lightSpotDirectionsSpot[spotLightCount], ref lightSpotEndsSpot[spotLightCount]))
                        {
                            spotLightCount++;
                        }
                        break;

                    default:
                        break;
                }
            }

            // dir lights
            Shader.SetGlobalInt("_WeatherMakerDirLightCount", dirLightCount);
            Shader.SetGlobalVectorArray("_WeatherMakerDirLightPosition", lightPositionsDir);
            Shader.SetGlobalVectorArray("_WeatherMakerDirLightColor", lightColorsDir);
            Shader.SetGlobalVectorArray("_WeatherMakerDirLightAtten", lightAttenDir);

            // point lights
            Shader.SetGlobalInt("_WeatherMakerPointLightCount", pointLightCount);
            Shader.SetGlobalVectorArray("_WeatherMakerPointLightPosition", lightPositionsPoint);
            Shader.SetGlobalVectorArray("_WeatherMakerPointLightColor", lightColorsPoint);
            Shader.SetGlobalVectorArray("_WeatherMakerPointLightAtten", lightAttenPoint);

            // spot lights
            Shader.SetGlobalInt("_WeatherMakerSpotLightCount", spotLightCount);
            Shader.SetGlobalVectorArray("_WeatherMakerSpotLightPosition", lightPositionsSpot);
            Shader.SetGlobalVectorArray("_WeatherMakerSpotLightColor", lightColorsSpot);
            Shader.SetGlobalVectorArray("_WeatherMakerSpotLightAtten", lightAttenSpot);
            Shader.SetGlobalVectorArray("_WeatherMakerSpotLightSpotDirection", lightSpotDirectionsSpot);
            Shader.SetGlobalVectorArray("_WeatherMakerSpotLightSpotEnd", lightSpotEndsSpot);
        }

        private void SetAllLightsToShader(Camera camera)
        {
            int lightCount;
            int lightIndex;
            int? nonDirLightIndex = null;
            lights.Sort(LightSorter);
            for (lightCount = 0, lightIndex = 0; lightIndex < lights.Count && lightCount < MaximumLightCount; lightIndex++)
            {
                SetLightAtIndex(lights[lightIndex], camera, ref lightCount, ref nonDirLightIndex);
            }

            // *** NOTE: if getting warnings about array sizes changing, simply restart the Unity editor ***

            // all lights
            Shader.SetGlobalVectorArray("_WeatherMakerLightPosition", lightPositions);
            Shader.SetGlobalVectorArray("_WeatherMakerLightColor", lightColors);
            Shader.SetGlobalVectorArray("_WeatherMakerLightAtten", lightAtten);
            Shader.SetGlobalVectorArray("_WeatherMakerLightSpotDirection", lightSpotDirections);
            Shader.SetGlobalVectorArray("_WeatherMakerLightSpotEnd", lightSpotEnds);
            Shader.SetGlobalInt("_WeatherMakerLightCount", lightCount);
            Shader.SetGlobalInt("_WeatherMakerNonDirLightIndex", (nonDirLightIndex == null ? lightCount : nonDirLightIndex.Value));

            // reduce lights by sun light and multiply by global point / spot multiplier
            float volumetricLightMultiplier = Mathf.Max(0.0f, (1.0f - (WeatherMakerScript.Instance.Sun.Light.intensity * FogLightSunIntensityReducer)));
            Shader.SetGlobalFloat("_WeatherMakerVolumetricPointSpotMultiplier", volumetricLightMultiplier);
        }

        private void Create3DNoiseTexture()
        {

#if UNITY_EDITOR

            /*
            TextAsset data = Resources.Load("WeatherMakerTextureFogNoise3D") as TextAsset;
            if (data == null)
            {
                return;
            }

            byte[] bytes = data.bytes;
            uint height = BitConverter.ToUInt32(data.bytes, 12);
            uint width = BitConverter.ToUInt32(data.bytes, 16);
            uint pitch = BitConverter.ToUInt32(data.bytes, 20);
            uint depth = BitConverter.ToUInt32(data.bytes, 24);
            uint formatFlags = BitConverter.ToUInt32(data.bytes, 20 * 4);
            uint fourCC = BitConverter.ToUInt32(data.bytes, 21 * 4);
            uint bitdepth = BitConverter.ToUInt32(data.bytes, 22 * 4);
            if (bitdepth == 0)
            {
                bitdepth = pitch / width * 8;
            }

            Texture3D t = new Texture3D((int)width, (int)height, (int)depth, TextureFormat.Alpha8, false);
            t.filterMode = FilterMode.Bilinear;
            t.wrapMode = TextureWrapMode.Repeat;
            t.name = "Noise 3D (Weather Maker)";

            Color32[] c = new Color32[width * height * depth];

            uint index = 128;
            if (data.bytes[21 * 4] == 'D' && data.bytes[21 * 4 + 1] == 'X' && data.bytes[21 * 4 + 2] == '1' &&
                data.bytes[21 * 4 + 3] == '0' && (formatFlags & 0x4) != 0)
            {
                uint format = BitConverter.ToUInt32(data.bytes, (int)index);
                if (format >= 60 && format <= 65)
                {
                    bitdepth = 8;
                }
                else if (format >= 48 && format <= 52)
                {
                    bitdepth = 16;
                }
                else if (format >= 27 && format <= 32)
                {
                    bitdepth = 32;
                }
                index += 20;
            }

            uint byteDepth = bitdepth / 8;
            pitch = (width * bitdepth + 7) / 8;

            for (int d = 0; d < depth; ++d)
            {
                for (int h = 0; h < height; ++h)
                {
                    for (int w = 0; w < width; ++w)
                    {
                        byte v = bytes[index + w * byteDepth];
                        c[w + h * width + d * width * height] = new Color32(v, v, v, v);
                    }

                    index += pitch;
                }
            }

            t.SetPixels32(c);
            t.Apply();
            */

            // Example code loading a file of raw bytes
            /*
            TextAsset textAsset = (TextAsset)Resources.Load("WeatherMakerTexturePerlinNoise3D");
            byte[] bytes = textAsset.bytes;
            if (bytes.Length != 256 * 256 * 256)
            {
                return;
            }
            Texture3D t = new Texture3D(256, 256, 256, TextureFormat.Alpha8, false);
            t.filterMode = FilterMode.Bilinear;
            t.wrapMode = TextureWrapMode.Repeat;
            t.name = "Perlin Noise 3D (Weather Maker)";
            Color32[] colors = new Color32[bytes.Length];
            byte v;
            for (int i = 0; i < colors.Length; i++)
            {
                v = bytes[i];
                colors[i] = new Color32(v, v, v, v);
            }
            t.SetPixels32(colors);
            t.Apply();
            */

            // UnityEditor.AssetDatabase.CreateAsset(t, "Assets/WeatherMakerTextureFogNoise3D.asset");

#endif

        }

        private void UpdateAllLights()
        {
            // if no user lights specified, find all the lights in the scene and sort them
            if (AutoFindLights)
            {
                Light[] allLights = GameObject.FindObjectsOfType<Light>();
                lights.Clear();
                foreach (Light light in allLights)
                {
                    if (light.enabled)
                    {
                        lights.Add(light);
                    }
                }
            }
            else
            {
                // add the sun if it is on, else remove it
                if (WeatherMakerScript.Instance.Sun.LightIsOn)
                {
                    AddLight(WeatherMakerScript.Instance.Sun.Light);
                }
                else
                {
                    RemoveLight(WeatherMakerScript.Instance.Sun.Light);
                }

                // add each moon if it is on, else remove it
                foreach (WeatherMakerCelestialObject moon in WeatherMakerScript.Instance.Moons)
                {
                    if (moon.LightIsOn)
                    {
                        AddLight(moon.Light);
                    }
                    else
                    {
                        RemoveLight(moon.Light);
                    }
                }

                // add each auto-add light if it is on, else remove it
                for (int i = AutoAddLights.Count - 1; i >= 0; i--)
                {
                    Light light = AutoAddLights[i];
                    if (light == null)
                    {
                        AutoAddLights.RemoveAt(i);
                    }
                    else if (light.intensity == 0.0f || !light.enabled || !light.gameObject.activeInHierarchy)
                    {
                        RemoveLight(light);
                    }
                    else
                    {
                        AddLight(light);
                    }
                }
            }
        }

        private void UpdateNullFogZones()
        {
            int nullFogZoneCount = 0;
            FogNullZones.Sort(NullFogZoneSorter);
            for (int i = 0; i < FogNullZones.Count && nullFogZoneCount < MaximumNullFogZones; i++)
            {
                if (GeometryUtility.TestPlanesAABB(WeatherMakerScript.Instance.CurrentCameraFrustumPlanes, FogNullZones[i]))
                {
                    nullFogZoneArrayMin[nullFogZoneCount] = FogNullZones[i].min;
                    nullFogZoneArrayMax[nullFogZoneCount] = FogNullZones[i].max;
                    nullFogZoneCount++;
                }
            }
            Shader.SetGlobalInt("_FogNullZoneCount", nullFogZoneCount);
            Shader.SetGlobalVectorArray("_FogNullZonesMin", nullFogZoneArrayMin);
            Shader.SetGlobalVectorArray("_FogNullZonesMax", nullFogZoneArrayMax);
            if (nullFogZoneCount == 0)
            {
                Shader.DisableKeyword("WEATHER_MAKER_FOG_ENABLE_NULL_FOG_ZONES");
            }
            else
            {
                Shader.EnableKeyword("WEATHER_MAKER_FOG_ENABLE_NULL_FOG_ZONES");
            }
        }

#if CREATE_DITHER_TEXTURE_FOR_WEATHER_MAKER_LIGHT_MANAGER

        private void CreateDitherTexture()
        {
            if (DitherTextureInstance != null)
            {
                return;
            }

#if DITHER_4_4

            int size = 4;

#else

            int size = 8;

#endif

            DitherTextureInstance = new Texture2D(size, size, TextureFormat.Alpha8, false, true);
            DitherTextureInstance.filterMode = FilterMode.Point;
            Color32[] c = new Color32[size * size];

            byte b;

#if DITHER_4_4

            b = (byte)(0.0f / 16.0f * 255); c[0] = new Color32(b, b, b, b);
            b = (byte)(8.0f / 16.0f * 255); c[1] = new Color32(b, b, b, b);
            b = (byte)(2.0f / 16.0f * 255); c[2] = new Color32(b, b, b, b);
            b = (byte)(10.0f / 16.0f * 255); c[3] = new Color32(b, b, b, b);

            b = (byte)(12.0f / 16.0f * 255); c[4] = new Color32(b, b, b, b);
            b = (byte)(4.0f / 16.0f * 255); c[5] = new Color32(b, b, b, b);
            b = (byte)(14.0f / 16.0f * 255); c[6] = new Color32(b, b, b, b);
            b = (byte)(6.0f / 16.0f * 255); c[7] = new Color32(b, b, b, b);

            b = (byte)(3.0f / 16.0f * 255); c[8] = new Color32(b, b, b, b);
            b = (byte)(11.0f / 16.0f * 255); c[9] = new Color32(b, b, b, b);
            b = (byte)(1.0f / 16.0f * 255); c[10] = new Color32(b, b, b, b);
            b = (byte)(9.0f / 16.0f * 255); c[11] = new Color32(b, b, b, b);

            b = (byte)(15.0f / 16.0f * 255); c[12] = new Color32(b, b, b, b);
            b = (byte)(7.0f / 16.0f * 255); c[13] = new Color32(b, b, b, b);
            b = (byte)(13.0f / 16.0f * 255); c[14] = new Color32(b, b, b, b);
            b = (byte)(5.0f / 16.0f * 255); c[15] = new Color32(b, b, b, b);

#else

            int i = 0;
            b = (byte)(1.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(49.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(13.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(61.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(4.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(52.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(16.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(64.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);

            b = (byte)(33.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(17.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(45.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(29.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(36.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(20.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(48.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(32.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);

            b = (byte)(9.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(57.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(5.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(53.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(12.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(60.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(8.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(56.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);

            b = (byte)(41.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(25.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(37.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(21.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(44.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(28.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(40.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(24.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);

            b = (byte)(3.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(51.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(15.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(63.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(2.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(50.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(14.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(62.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);

            b = (byte)(35.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(19.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(47.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(31.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(34.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(18.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(46.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(30.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);

            b = (byte)(11.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(59.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(7.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(55.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(10.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(58.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(6.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(54.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);

            b = (byte)(43.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(27.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(39.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(23.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(42.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(26.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(38.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
            b = (byte)(22.0f / 65.0f * 255); c[i++] = new Color32(b, b, b, b);
#endif

            DitherTextureInstance.SetPixels32(c);
            DitherTextureInstance.Apply();
        }

#endif

        private void Awake()
        {
            Instance = this;
            FogNullZones = new List<Bounds>();
            // Create3DNoiseTexture();
            // CreateDitherTexture();
            NoiseTexture3DInstance = NoiseTexture3D;
            Shader.SetGlobalTexture("_WeatherMakerNoiseTexture3D", NoiseTexture3D);
        }

        private void Update()
        {
            FogNullZones.Clear();
        }

        private void LateUpdate()
        {
            Shader.SetGlobalFloat("_WeatherMakerFogDirectionalLightScatterIntensity", FogDirectionalLightScatterIntensity);
            Shader.SetGlobalVector("_WeatherMakerFogLightFalloff", new Vector4(FogSpotLightRadiusFalloff, 0.0f, 0.0f, 0.0f));
            Shader.SetGlobalFloat("_WeatherMakerFogLightSunIntensityReducer", FogLightSunIntensityReducer);
            UpdateAllLights();
        }

        /// <summary>
        /// Add a light, unless AutoFindLights is true
        /// </summary>
        /// <param name="l">Light to add</param>
        /// <returns>True if light added, false if not</returns>
        public bool AddLight(Light l)
        {
            if (l != null && !AutoFindLights && !lights.Contains(l))
            {
                lights.Add(l);
                return true;
            }
            return false;
        }

        /// <summary>
        /// Remove a light, unless AutoFindLights is true
        /// </summary>
        /// <param name="l"></param>
        /// <returns>True if light removed, false if not</returns>
        public bool RemoveLight(Light l)
        {
            if (!AutoFindLights)
            {
                return lights.Remove(l);
            }
            return false;
        }

        /// <summary>
        /// Called when a camera is about to render - sets up shader and light properties, etc.
        /// </summary>
        /// <param name="camera">The current camera</param>
        public void PrepareToRenderCamera(Camera camera)
        {

#if UNITY_EDITOR

            if (!Application.isPlaying)
            {
                return;
            }

#endif

            // all lights, also populates lights list, sorted
            SetAllLightsToShader(camera);

            // add lights for each type
            SetLightsByTypeToShader(camera);

            // update null fog zones
            UpdateNullFogZones();
        }

        /// <summary>
        /// Current set of lights
        /// </summary>
        public IEnumerable<Light> Lights { get { return lights; } }

        /// <summary>
        /// Shared instance of light manager
        /// </summary>
        public static WeatherMakerLightManagerScript Instance { get; private set; }
    }
}