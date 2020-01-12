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
using UnityEngine.Rendering;

namespace DigitalRuby.WeatherMaker
{
    [ExecuteInEditMode]
    [RequireComponent(typeof(MeshRenderer))]
    public class WeatherMakerSkyPlaneScript : MonoBehaviour
    {
        [Header("Rendering")]
        [Tooltip("Sky plane material")]
        public Material Material;

        [Header("Night Sky")]
        [Tooltip("Night texture")]
        public Texture2D NightTexture;

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
        [Tooltip("Gradient for sky, usually to increase color change as sun nears horizon.")]
        public Gradient SkyGradient;

        [Range(0.0f, 1.0f)]
        [Tooltip("Dither level")]
        public float DitherLevel = 0.005f;

        [Tooltip("Sky plane y offset. Causes sky ray to increase, moving the sunset, sunrise and sky horizon up.")]
        [Range(0.0f, 0.5f)]
        public float SkyYOffset = 0.25f;

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

        private MeshRenderer meshRenderer;

#if UNITY_EDITOR

        [Header("Plane Generation")]
        [Tooltip("The number of rows in the plane. The higher the number, the higher quality sky.")]
        [Range(4, 96)]
        public int PlaneRows = 96;

        [SerializeField]
        [HideInInspector]
        private int lastPlaneRows = -1;

        [Tooltip("The number of columns in the sky plane. The higher the number, the higher quality sky.")]
        [Range(4, 96)]
        public int PlaneColumns = 32;

        [SerializeField]
        [HideInInspector]
        private int lastPlaneColumns = -1;

        private Mesh CreatePlaneMesh()
        {
            Mesh m = new Mesh { name = "WeatherMakerSkyPlane" };
            int vertIndex = 0;
            float stepX = 1.0f / (float)PlaneColumns;
            float stepY = 1.0f / (float)PlaneRows;
            List<Vector3> vertices = new List<Vector3>(PlaneColumns * PlaneRows);
            List<Vector2> uvs = new List<Vector2>(PlaneColumns * PlaneRows);
            List<int> triangles = new List<int>();
            for (float x = -0.5f; x < 0.5f; x += stepX)
            {
                for (float y = -0.5f; y < 0.5f; y += stepY)
                {
                    vertices.Add(new Vector3(x, y, 0.0f));
                    vertices.Add(new Vector3(x, y + stepY, 0.0f));
                    vertices.Add(new Vector3(x + stepX, y + stepY, 0.0f));
                    vertices.Add(new Vector3(x + stepX, y, 0.0f));
                    uvs.Add(new Vector2(x + 0.5f, y + 0.5f));
                    uvs.Add(new Vector2(x + 0.5f, y + stepY + 0.5f));
                    uvs.Add(new Vector2(x + stepX + 0.5f, y + stepY + 0.5f));
                    uvs.Add(new Vector2(x + stepX + 0.5f, y + 0.5f));
                    triangles.Add(vertIndex++);
                    triangles.Add(vertIndex++);
                    triangles.Add(vertIndex);
                    triangles.Add(vertIndex--);
                    triangles.Add(--vertIndex);
                    triangles.Add(vertIndex += 3);
                    vertIndex++;
                }
            }
            m.SetVertices(vertices);
            m.SetUVs(0, uvs);
            m.SetTriangles(triangles, 0);
            m.RecalculateNormals();
            return m;
        }

#endif

        private void UpdateShaderProperties(Material m)
        {
            bool gamma = (QualitySettings.activeColorSpace == ColorSpace.Gamma);
            m.SetTexture("_NightTex", NightTexture);
            Shader.SetGlobalFloat("_NightMultiplier", WeatherMakerScript.Instance.DayNightScript.NightMultiplier);
            Shader.SetGlobalFloat("_NightSkyMultiplier", Mathf.Max(1.0f - Mathf.Min(1.0f, SkyAtmosphereThickness), WeatherMakerScript.Instance.DayNightScript.NightMultiplier));
            Shader.SetGlobalFloat("_NightVisibilityThreshold", NightVisibilityThreshold);
            Shader.SetGlobalFloat("_NightIntensity", (gamma ? 2.0f * NightIntensity : NightIntensity));
            Shader.SetGlobalFloat("_WeatherMakerSkyDitherLevel", (gamma ? DitherLevel : DitherLevel * 0.5f));
            Shader.SetGlobalFloat("_WeatherMakerSkyYOffset", SkyYOffset);
            Material.DisableKeyword("ENABLE_NIGHT_TWINKLE");
            if (NightTwinkleRandomness > 0.0f || (NightTwinkleVariance > 0.0f && NightTwinkleSpeed > 0.0f))
            {
                Shader.SetGlobalFloat("_NightTwinkleSpeed", NightTwinkleSpeed);
                Shader.SetGlobalFloat("_NightTwinkleVariance", NightTwinkleVariance);
                Shader.SetGlobalFloat("_NightTwinkleMinimum", NightTwinkleMinimum);
                Shader.SetGlobalFloat("_NightTwinkleRandomness", NightTwinkleRandomness);
                Material.EnableKeyword("ENABLE_NIGHT_TWINKLE");
            }
            WeatherMakerSkySphereScript.SetGlobalSkyParameters(SkyAtmosphereMie, SkyMieMultiplier, SkyRayleighMultiplier, SkyAtmosphereThickness, SkyTintColor,
                SkyOuterRadius, SkyInnerRadius, SkyCameraHeight);
        }

        private void UpdateSkyPlane(Camera c)
        {
            UpdateShaderProperties(Material);

            float pos = (c.nearClipPlane + 0.1f);
            Vector3 topLeft = c.ViewportToWorldPoint(Vector3.zero);
            Vector3 bottomRight = c.ViewportToWorldPoint(Vector3.one);
            transform.position = c.transform.position + (c.transform.forward * pos);
            transform.localScale = new Vector3(1.01f * (bottomRight.x - topLeft.x), 1.01f * (bottomRight.y - topLeft.y), 1.0f);
        }

        private void Awake()
        {
            meshRenderer = GetComponent<MeshRenderer>();
            meshRenderer.sortingOrder = int.MinValue;
        }

        private void Update()
        {

#if UNITY_EDITOR

            if (!Application.isPlaying)
            {
                PreCullCamera(WeatherMakerScript.Instance.Camera);
            }
            if (GetComponent<MeshFilter>().sharedMesh == null || lastPlaneColumns != PlaneColumns || lastPlaneRows != PlaneRows)
            {
                lastPlaneColumns = PlaneColumns;
                lastPlaneRows = PlaneRows;
                GetComponent<MeshFilter>().sharedMesh = CreatePlaneMesh();
            }

#endif

        }

        public void PreCullCamera(Camera c)
        {
            if (c == null)
            {
                return;
            }
            UpdateSkyPlane(c);
        }
    }
}
