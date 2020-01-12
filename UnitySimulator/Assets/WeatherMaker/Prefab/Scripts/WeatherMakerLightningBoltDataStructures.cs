//
// Procedural Lightning for Unity
// (c) 2015 Digital Ruby, LLC
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

// uncomment to enable profiling using stopwatch and debug.log
// #define ENABLE_PROFILING

#if UNITY_4_0 || UNITY_4_1 || UNITY_4_2 || UNITY_4_3 || UNITY_4_4 || UNITY_4_5 || UNITY_4_6 || UNITY_4_7 || UNITY_4_8 || UNITY_4_9

#define UNITY_4

#endif

#if UNITY_4 || UNITY_5_0 || UNITY_5_1 || UNITY_5_2

#define UNITY_PRE_5_3

#endif

#if NETFX_CORE

#define TASK_AVAILABLE

using System.Threading.Tasks;

#endif

using UnityEngine;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;

namespace DigitalRuby.WeatherMaker
{
    /// <summary>
    /// Quality settings for lightning
    /// </summary>
    public enum LightningBoltQualitySetting
    {
        /// <summary>
        /// Use all settings from the script, ignoring the global quality setting
        /// </summary>
        UseScript,

        /// <summary>
        /// Use the global quality setting to determine lightning quality and maximum number of lights and shadowing
        /// </summary>
        LimitToQualitySetting
    }

    /// <summary>
    /// Camera modes
    /// </summary>
    public enum CameraMode
    {
        /// <summary>
        /// Auto detect
        /// </summary>
        Auto,

        /// <summary>
        /// Force perspective camera lightning
        /// </summary>
        Perspective,

        /// <summary>
        /// Force orthographic XY lightning
        /// </summary>
        OrthographicXY,

        /// <summary>
        /// Force orthographic XZ lightning
        /// </summary>
        OrthographicXZ,

        /// <summary>
        /// Unknown camera mode (do not use)
        /// </summary>
        Unknown
    }

    [System.Serializable]
    public class LightningLightParameters
    {
        /// <summary>
        /// Light render mode
        /// </summary>
        [Tooltip("Light render mode - leave as auto unless you have special use cases")]
        [HideInInspector]
        public LightRenderMode RenderMode = LightRenderMode.Auto;

        /// <summary>
        /// Color of light
        /// </summary>
        [Tooltip("Color of the light")]
        public Color LightColor = Color.white;

        /// <summary>
        /// What percent of segments should have a light? Keep this pretty low for performance, i.e. 0.05 or lower depending on generations
        /// Set really really low to only have 1 light, i.e. 0.0000001f
        /// For example, at generations 5, the main trunk has 32 segments, 64 at generation 6, etc.
        /// If non-zero, there wil be at least one light in the middle
        /// </summary>
        [Tooltip("What percent of segments should have a light? For performance you may want to keep this small.")]
        [Range(0.0f, 1.0f)]
        public float LightPercent = 0.000001f;

        /// <summary>
        /// What percent of lights created should cast shadows?
        /// </summary>
        [Tooltip("What percent of lights created should cast shadows?")]
        [Range(0.0f, 1.0f)]
        public float LightShadowPercent;

        /// <summary>
        /// Light intensity
        /// </summary>
        [Tooltip("Light intensity")]
        [Range(0.0f, 8.0f)]
        public float LightIntensity = 0.5f;

        /// <summary>
        /// Bounce intensity
        /// </summary>
        [Tooltip("Bounce intensity")]
        [Range(0.0f, 8.0f)]
        public float BounceIntensity;

        /// <summary>
        /// Shadow strength, 0 - 1. 0 means all light, 1 means all shadow
        /// </summary>
        [Tooltip("Shadow strength, 0 means all light, 1 means all shadow")]
        [Range(0.0f, 1.0f)]
        public float ShadowStrength = 1.0f;

        /// <summary>
        /// Shadow bias
        /// </summary>
        [Tooltip("Shadow bias, 0 - 2")]
        [Range(0.0f, 2.0f)]
        public float ShadowBias = 0.05f;

        /// <summary>
        /// Shadow normal bias
        /// </summary>
        [Tooltip("Shadow normal bias, 0 - 3")]
        [Range(0.0f, 3.0f)]
        public float ShadowNormalBias = 0.4f;

        /// <summary>
        /// Light range
        /// </summary>
        [Tooltip("The range of each light created")]
        public float LightRange;

        /// <summary>
        /// Only light up objects that match this layer mask
        /// </summary>
        [Tooltip("Only light objects that match this layer mask")]
        public LayerMask CullingMask = ~0;

        /// <summary>
        /// Should light be shown for these parameters?
        /// </summary>
        public bool HasLight
        {
            get { return (LightColor.a > 0.0f && LightIntensity >= 0.01f && LightPercent >= 0.0000001f && LightRange > 0.01f); }
        }
    }

    /// <summary>
    /// Parameters that control lightning bolt behavior
    /// </summary>
    [System.Serializable]
    public sealed class LightningBoltParameters
    {
        #region Internal use only

        // INTERNAL USE ONLY!!!
        private static int randomSeed = Environment.TickCount;
        private static readonly List<LightningBoltParameters> cache = new List<LightningBoltParameters>();
        internal int generationWhereForksStop;
        internal int forkednessCalculated;
        internal LightningBoltQualitySetting quality;
        internal float delaySeconds;
        internal int maxLights;
        // END INTERNAL USE ONLY

        #endregion Internal use only

        /// <summary>
        /// Scale all scalar parameters by this value (i.e. trunk width, turbulence, turbulence velocity)
        /// </summary>
        public static float Scale = 1.0f;

        /// <summary>
        /// Contains quality settings for different quality levels. By default, this assumes 6 quality levels, so if you have your own
        /// custom quality setting levels, you may want to clear this dictionary out and re-populate it with your own limits
        /// </summary>
        public static readonly Dictionary<int, LightningQualityMaximum> QualityMaximums = new Dictionary<int, LightningQualityMaximum>();

        static LightningBoltParameters()
        {
            string[] names = QualitySettings.names;
            for (int i = 0; i < names.Length; i++)
            {
                switch (i)
                {
                    case 0:
                        QualityMaximums[i] = new LightningQualityMaximum { MaximumGenerations = 3, MaximumLightPercent = 0, MaximumShadowPercent = 0.0f };
                        break;
                    case 1:
                        QualityMaximums[i] = new LightningQualityMaximum { MaximumGenerations = 4, MaximumLightPercent = 0, MaximumShadowPercent = 0.0f };
                        break;
                    case 2:
                        QualityMaximums[i] = new LightningQualityMaximum { MaximumGenerations = 5, MaximumLightPercent = 0.1f, MaximumShadowPercent = 0.0f };
                        break;
                    case 3:
                        QualityMaximums[i] = new LightningQualityMaximum { MaximumGenerations = 5, MaximumLightPercent = 0.1f, MaximumShadowPercent = 0.0f };
                        break;
                    case 4:
                        QualityMaximums[i] = new LightningQualityMaximum { MaximumGenerations = 6, MaximumLightPercent = 0.05f, MaximumShadowPercent = 0.1f };
                        break;
                    case 5:
                        QualityMaximums[i] = new LightningQualityMaximum { MaximumGenerations = 7, MaximumLightPercent = 0.025f, MaximumShadowPercent = 0.05f };
                        break;
                    default:
                        QualityMaximums[i] = new LightningQualityMaximum { MaximumGenerations = 8, MaximumLightPercent = 0.025f, MaximumShadowPercent = 0.05f };
                        break;
                }
            }
        }

        /// <summary>
        /// Constructor
        /// </summary>
        public LightningBoltParameters()
        {
            unchecked
            {
                random = currentRandom = new System.Random(randomSeed++);
            }
        }

        /// <summary>
        /// Generator to create the lightning bolt from the parameters
        /// </summary>
        public LightningGenerator Generator;

        /// <summary>
        /// Start of the bolt
        /// </summary>
        public Vector3 Start;

        /// <summary>
        /// End of the bolt
        /// </summary>
        public Vector3 End;

        /// <summary>
        /// X, Y and Z radius variance from Start
        /// </summary>
        public Vector3 StartVariance;

        /// <summary>
        /// X, Y and Z radius variance from End
        /// </summary>
        public Vector3 EndVariance;

        private int generations;
        /// <summary>
        /// Number of generations (0 for just a point light, otherwise 1 - 8). Higher generations have lightning with finer detail but more expensive to create.
        /// </summary>
        public int Generations
        {
            get { return generations; }
            set
            {
                int v = Mathf.Clamp(value, 1, 8);

                if (quality == LightningBoltQualitySetting.UseScript)
                {
                    generations = v;
                }
                else
                {
                    LightningQualityMaximum maximum;
                    int level = QualitySettings.GetQualityLevel();
                    if (QualityMaximums.TryGetValue(level, out maximum))
                    {
                        generations = Mathf.Min(maximum.MaximumGenerations, v);
                    }
                    else
                    {
                        generations = v;
                        Debug.LogError("Unable to read lightning quality settings from level " + level.ToString());
                    }
                }
            }
        }

        /// <summary>
        /// How long the bolt should live in seconds
        /// </summary>
        public float LifeTime;

        /// <summary>
        /// Minimum delay
        /// </summary>
        public float Delay;

        /// <summary>
        /// How long to wait in seconds before starting additional lightning bolts
        /// </summary>
        public RangeOfFloats DelayRange;

        /// <summary>
        /// How chaotic is the lightning? (0 - 1). Higher numbers create more chaotic lightning.
        /// </summary>
        public float ChaosFactor;

        /// <summary>
        /// The width of the trunk
        /// </summary>
        public float TrunkWidth;

        /// <summary>
        /// The ending width of a segment of lightning
        /// </summary>
        public float EndWidthMultiplier = 0.5f;

        /// <summary>
        /// Intensity of the lightning
        /// </summary>
        public float Intensity = 1.0f;

        /// <summary>
        /// Intensity of the glow
        /// </summary>
        public float GlowIntensity;

        /// <summary>
        /// Glow width multiplier
        /// </summary>
        public float GlowWidthMultiplier;

        /// <summary>
        /// How forked the lightning should be, 0 for none, 1 for LOTS of forks
        /// </summary>
        public float Forkedness;

        /// <summary>
        /// This is subtracted from the initial generations value, and any generation below that cannot have a fork
        /// </summary>
        public int GenerationWhereForksStopSubtractor = 5;

        /// <summary>
        /// Used to generate random numbers. Not thread safe.
        /// </summary>
        public System.Random Random
        {
            get { return currentRandom; }
            set
            {
                random = value ?? random;
                currentRandom = (randomOverride ?? random);
            }
        }
        private System.Random random;
        private System.Random currentRandom;

        /// <summary>
        /// Override Random to a different Random. This gets set back to null when the parameters go back to the cache. Great for a one time bolt that looks a certain way.
        /// </summary>
        public System.Random RandomOverride
        {
            get { return randomOverride; }
            set
            {
                randomOverride = value;
                currentRandom = (randomOverride ?? random);
            }
        }
        private System.Random randomOverride;

        /// <summary>
        /// The percent of time the lightning should fade in and out (0 - 1). Example: 0.2 would fade in for 20% of the lifetime and fade out for 20% of the lifetime. Set to 0 for no fade.
        /// </summary>
        public float FadePercent;

        private float growthMultiplier;
        /// <summary>
        /// A value between 0 and 0.999 that determines how fast the lightning should grow over the lifetime. A value of 1 grows slowest, 0 grows instantly
        /// </summary>
        public float GrowthMultiplier
        {
            get { return growthMultiplier; }
            set { growthMultiplier = Mathf.Clamp(value, 0.0f, 0.999f); }
        }

        /// <summary>
        /// Minimum distance multiplier for forks
        /// </summary>
        public float ForkLengthMultiplier = 0.6f;

        /// <summary>
        /// Variance of the fork distance (random range of 0 to n is added to ForkLengthMultiplier)
        /// </summary>
        public float ForkLengthVariance = 0.2f;

        /// <summary>
        /// Forks will have their end widths multiplied by this value
        /// </summary>
        public float ForkEndWidthMultiplier = 1.0f;

        /// <summary>
        /// Light parameters, null for none
        /// </summary>
        public LightningLightParameters LightParameters;

        /// <summary>
        /// Points for the trunk to follow - not all generators support this
        /// </summary>
        public List<Vector3> Points { get; set; }

        /// <summary>
        /// The amount of smoothing applied. For example, if there were 4 original points and smoothing / spline created 32 points, this value would be 8 - not all generators support this
        /// </summary>
        public int SmoothingFactor;

        /// <summary>
        /// Get a multiplier for fork distance
        /// </summary>
        /// <returns>Fork multiplier</returns>
        public float ForkMultiplier()
        {
            return ((float)Random.NextDouble() * ForkLengthVariance) + ForkLengthMultiplier;
        }

        /// <summary>
        /// Apply variance to a vector
        /// </summary>
        /// <param name="pos">Position</param>
        /// <param name="variance">Variance</param>
        /// <returns>New position</returns>
        public Vector3 ApplyVariance(Vector3 pos, Vector3 variance)
        {
            return new Vector3
            (
                pos.x + (((float)Random.NextDouble() * 2.0f) - 1.0f) * variance.x,
                pos.y + (((float)Random.NextDouble() * 2.0f) - 1.0f) * variance.y,
                pos.z + (((float)Random.NextDouble() * 2.0f) - 1.0f) * variance.z
            );
        }

        /// <summary>
        /// Get or create lightning bolt parameters. If cache has parameters, one is taken, otherwise a new object is created. NOT thread safe.
        /// </summary>
        /// <returns>Lightning bolt parameters</returns>
        public static LightningBoltParameters GetOrCreateParameters()
        {
            LightningBoltParameters p;
            if (cache.Count == 0)
            {
                unchecked
                {
                    p = new LightningBoltParameters();
                }
            }
            else
            {
                int i = cache.Count - 1;
                p = cache[i];
                cache.RemoveAt(i);
            }
            return p;
        }

        /// <summary>
        /// Return parameters to cache. NOT thread safe.
        /// </summary>
        /// <param name="p">Parameters</param>
        public static void ReturnParametersToCache(LightningBoltParameters p)
        {
            if (!cache.Contains(p))
            {
                // reset variables that are state-machine dependant
                p.Start = p.End = Vector3.zero;
                p.Points = null;
                p.Generator = null;
                p.SmoothingFactor = 0;
                p.RandomOverride = null;
                cache.Add(p);
            }
        }
    }

    /// <summary>
    /// A group of lightning bolt segments, such as the main trunk of the lightning bolt
    /// </summary>
    public class LightningBoltSegmentGroup
    {
        /// <summary>
        /// Width
        /// </summary>
        public float LineWidth;

        /// <summary>
        /// Start index of the segment to render (for performance, some segments are not rendered and only used for calculations)
        /// </summary>
        public int StartIndex;

        /// <summary>
        /// Generation
        /// </summary>
        public int Generation;

        /// <summary>
        /// Delay before rendering should start
        /// </summary>
        public float Delay;

        /// <summary>
        /// Peak start, the segments should be fully visible at this point
        /// </summary>
        public float PeakStart;

        /// <summary>
        /// Peak end, the segments should start to go away after this point
        /// </summary>
        public float PeakEnd;

        /// <summary>
        /// Total life time the group will be alive in seconds
        /// </summary>
        public float LifeTime;

        /// <summary>
        /// The width can be scaled down to the last segment by this amount if desired
        /// </summary>
        public float EndWidthMultiplier;

        /// <summary>
        /// Color for the group
        /// </summary>
        public Color32 Color;

        /// <summary>
        /// Total number of active segments
        /// </summary>
        public int SegmentCount { get { return Segments.Count - StartIndex; } }

        /// <summary>
        /// Segments
        /// </summary>
        public readonly List<LightningBoltSegment> Segments = new List<LightningBoltSegment>();

        /// <summary>
        /// Lights
        /// </summary>
        public readonly List<Light> Lights = new List<Light>();

        /// <summary>
        /// Light parameters
        /// </summary>
        public LightningLightParameters LightParameters;

        /// <summary>
        /// Return the group to its cache if there is one
        /// </summary>
        public void Reset()
        {
            LightParameters = null;
            Segments.Clear();
            Lights.Clear();
            StartIndex = 0;
        }
    }

    /// <summary>
    /// A single segment of a lightning bolt
    /// </summary>
    public struct LightningBoltSegment
    {
        public Vector3 Start;
        public Vector3 End;

        public override string ToString()
        {
            return Start.ToString() + ", " + End.ToString();
        }
    }

    /// <summary>
    /// Contains maximum values for a given quality settings
    /// </summary>
    public class LightningQualityMaximum
    {
        /// <summary>
        /// Maximum generations
        /// </summary>
        public int MaximumGenerations { get; set; }

        /// <summary>
        /// Maximum light percent
        /// </summary>
        public float MaximumLightPercent { get; set; }

        /// <summary>
        /// Maximum light shadow percent
        /// </summary>
        public float MaximumShadowPercent { get; set; }
    }

    /// <summary>
    /// Lightning bolt dependencies
    /// </summary>
    public class LightningBoltDependencies
    {
        /// <summary>
        /// Parent - do not access from threads
        /// </summary>
        public GameObject Parent;

        /// <summary>
        /// Material - do not access from threads
        /// </summary>
        public Material LightningMaterialMesh;

        /// <summary>
        /// Material no glow - do not access from threads
        /// </summary>
        public Material LightningMaterialMeshNoGlow;

        /// <summary>
        /// Origin particle system - do not access from threads
        /// </summary>
        public ParticleSystem OriginParticleSystem;

        /// <summary>
        /// Dest particle system - do not access from threads
        /// </summary>
        public ParticleSystem DestParticleSystem;

        /// <summary>
        /// Camera position
        /// </summary>
        public Vector3 CameraPos;

        /// <summary>
        /// Is camera 2D?
        /// </summary>
        public bool CameraIsOrthographic;

        /// <summary>
        /// Camera mode
        /// </summary>
        public CameraMode CameraMode;

        /// <summary>
        /// Use world space
        /// </summary>
        public bool UseWorldSpace;

        /// <summary>
        /// Sort layer name
        /// </summary>
        public string SortLayerName;

        /// <summary>
        /// Order in layer
        /// </summary>
        public int SortOrderInLayer;

        /// <summary>
        /// Parameters
        /// </summary>
        public ICollection<LightningBoltParameters> Parameters;

#if !UNITY_WEBGL

        /// <summary>
        /// Thread state
        /// </summary>
        public LightningThreadState ThreadState;

#endif

        /// <summary>
        /// Method to start co-routines
        /// </summary>
        public Func<IEnumerator, Coroutine> StartCoroutine;

        /// <summary>
        /// Call this when a light is added
        /// </summary>
        public System.Action<Light> LightAdded;

        /// <summary>
        /// Call this when a light is removed
        /// </summary>
        public System.Action<Light> LightRemoved;

        /// <summary>
        /// Call this when the bolt becomes active
        /// </summary>
        public System.Action<LightningBolt> AddActiveBolt;

        /// <summary>
        /// Returns the dependencies to their cache
        /// </summary>
        public System.Action<LightningBoltDependencies> ReturnToCache;

        /// <summary>
        /// Runs when a lightning bolt is started (parameters, start, end)
        /// </summary>
        public System.Action<LightningBoltParameters, Vector3, Vector3> LightningBoltStarted;

        /// <summary>
        /// Runs when a lightning bolt is ended (parameters, start, end)
        /// </summary>
        public System.Action<LightningBoltParameters, Vector3, Vector3> LightningBoltEnded;
    }

    /// <summary>
    /// Lightning bolt
    /// </summary>
    public class LightningBolt
    {
        #region LineRendererMesh

        /// <summary>
        /// Class the encapsulates a game object, and renderer for lightning bolt meshes
        /// </summary>
        public class LineRendererMesh
        {
            #region Public variables

            public GameObject GameObject { get; private set; }

            public Material Material
            {
                get { return meshRenderer.sharedMaterial; }
                set { meshRenderer.sharedMaterial = value; }
            }

            public MeshRenderer MeshRenderer
            {
                get { return meshRenderer; }
            }

            public int Tag { get; set; }

            #endregion Public variables

            #region Public methods

            public LineRendererMesh()
            {
                GameObject = new GameObject("LightningBoltMeshRenderer");
                GameObject.SetActive(false); // call Begin to activate

#if UNITY_EDITOR

                GameObject.hideFlags = HideFlags.HideInHierarchy | HideFlags.HideInInspector;

#endif

                mesh = new Mesh { name = "ProceduralLightningMesh" };
                mesh.MarkDynamic();
                meshFilter = GameObject.AddComponent<MeshFilter>();
                meshFilter.sharedMesh = mesh;
                meshRenderer = GameObject.AddComponent<MeshRenderer>();

#if !UNITY_4

                meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                meshRenderer.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;

#endif

#if UNITY_5_3_OR_NEWER

                meshRenderer.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;

#endif

                meshRenderer.receiveShadows = false;
            }

            public void PopulateMesh()
            {

#if ENABLE_PROFILING

                System.Diagnostics.Stopwatch w = System.Diagnostics.Stopwatch.StartNew();

#endif

                if (vertices.Count == 0)
                {
                    mesh.Clear();
                }
                else
                {
                    PopulateMeshInternal();
                }

#if ENABLE_PROFILING

                Debug.LogFormat("MESH: {0}", w.Elapsed.TotalMilliseconds);

#endif

            }

            public bool PrepareForLines(int lineCount)
            {
                int vertexCount = lineCount * 4;
                if (vertices.Count + vertexCount > 64999)
                {
                    return false;
                }
                return true;
            }

            public void BeginLine(Vector3 start, Vector3 end, float radius, Color32 color, float colorIntensity, Vector4 fadeLifeTime, float glowWidthModifier, float glowIntensity)
            {
                Vector4 dir = (end - start);
                dir.w = radius;
                AppendLineInternal(ref start, ref end, ref dir, ref dir, ref dir, color, colorIntensity, ref fadeLifeTime, glowWidthModifier, glowIntensity);
            }

            public void AppendLine(Vector3 start, Vector3 end, float radius, Color32 color, float colorIntensity, Vector4 fadeLifeTime, float glowWidthModifier, float glowIntensity)
            {
                Vector4 dir = (end - start);
                dir.w = radius;
                Vector4 dirPrev1 = lineDirs[lineDirs.Count - 3];
                Vector4 dirPrev2 = lineDirs[lineDirs.Count - 1];
                AppendLineInternal(ref start, ref end, ref dir, ref dirPrev1, ref dirPrev2, color, colorIntensity, ref fadeLifeTime, glowWidthModifier, glowIntensity);
            }

            public void Reset()
            {
                Tag++;
                GameObject.SetActive(false);
                mesh.Clear();
                indices.Clear();
                vertices.Clear();
                colors.Clear();
                lineDirs.Clear();
                ends.Clear();

#if UNITY_PRE_5_3

				texCoords.Clear();
				glowModifiers.Clear();
				fadeXY.Clear();
				fadeZW.Clear();

#else

                texCoordsAndGlowModifiers.Clear();
                fadeLifetimes.Clear();

#endif

                currentBoundsMaxX = currentBoundsMaxY = currentBoundsMaxZ = int.MinValue + boundsPadder;
                currentBoundsMinX = currentBoundsMinY = currentBoundsMinZ = int.MaxValue - boundsPadder;
            }

            #endregion Public methods

            #region Private variables

            private static readonly Vector2 uv1 = new Vector2(0.0f, 0.0f);
            private static readonly Vector2 uv2 = new Vector2(1.0f, 0.0f);
            private static readonly Vector2 uv3 = new Vector2(0.0f, 1.0f);
            private static readonly Vector2 uv4 = new Vector2(1.0f, 1.0f);

            private readonly List<int> indices = new List<int>();

            private readonly List<Vector3> vertices = new List<Vector3>();
            private readonly List<Vector4> lineDirs = new List<Vector4>();
            private readonly List<Color32> colors = new List<Color32>();
            private readonly List<Vector3> ends = new List<Vector3>();

#if UNITY_PRE_5_3

            private readonly List<Vector2> texCoords = new List<Vector2>();
			private readonly List<Vector2> glowModifiers = new List<Vector2>();
			private readonly List<Vector2> fadeXY = new List<Vector2>();
			private readonly List<Vector2> fadeZW = new List<Vector2>();

#else

            private readonly List<Vector4> texCoordsAndGlowModifiers = new List<Vector4>();
            private readonly List<Vector4> fadeLifetimes = new List<Vector4>();

#endif

            private const int boundsPadder = 1000000000;
            private int currentBoundsMinX = int.MaxValue - boundsPadder;
            private int currentBoundsMinY = int.MaxValue - boundsPadder;
            private int currentBoundsMinZ = int.MaxValue - boundsPadder;
            private int currentBoundsMaxX = int.MinValue + boundsPadder;
            private int currentBoundsMaxY = int.MinValue + boundsPadder;
            private int currentBoundsMaxZ = int.MinValue + boundsPadder;

            private Mesh mesh;
            private MeshFilter meshFilter;
            private MeshRenderer meshRenderer;

            #endregion Private variables

            #region Private methods

            private void PopulateMeshInternal()
            {
                GameObject.SetActive(true);

#if UNITY_PRE_5_3

                mesh.vertices = vertices.ToArray();
                mesh.tangents = lineDirs.ToArray();
                mesh.colors32 = colors.ToArray();
				mesh.uv = texCoords.ToArray();
				mesh.uv2 = glowModifiers.ToArray();

// Unity 5.0 - 5.2.X has to use uv3 and uv4
// Unity 4.X does not support glow or fade or elapsed time
#if UNITY_5_0 || UNITY_5_1 || UNITY_5_2

				mesh.uv3 = fadeXY.ToArray();
				mesh.uv4 = fadeZW.ToArray();

#endif

                mesh.normals = ends.ToArray();
                mesh.triangles = indices.ToArray();

#else

                mesh.SetVertices(vertices);
                mesh.SetTangents(lineDirs);
                mesh.SetColors(colors);
                mesh.SetUVs(0, texCoordsAndGlowModifiers);
                mesh.SetUVs(1, fadeLifetimes);
                mesh.SetNormals(ends);
                mesh.SetTriangles(indices, 0);

#endif

                Bounds b = new Bounds();
                Vector3 min = new Vector3(currentBoundsMinX - 2, currentBoundsMinY - 2, currentBoundsMinZ - 2);
                Vector3 max = new Vector3(currentBoundsMaxX + 2, currentBoundsMaxY + 2, currentBoundsMaxZ + 2);
                b.center = (max + min) * 0.5f;
                b.size = (max - min) * 1.2f;
                mesh.bounds = b;
            }

            private void UpdateBounds(ref Vector3 point1, ref Vector3 point2)
            {
                // r = y + ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))); // min(x, y)
                // r = x - ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))); // max(x, y)

                unchecked
                {
                    {
                        int xCalculation = (int)point1.x - (int)point2.x;
                        xCalculation &= (xCalculation >> 31);
                        int xMin = (int)point2.x + xCalculation;
                        int xMax = (int)point1.x - xCalculation;

                        xCalculation = currentBoundsMinX - xMin;
                        xCalculation &= (xCalculation >> 31);
                        currentBoundsMinX = xMin + xCalculation;

                        xCalculation = currentBoundsMaxX - xMax;
                        xCalculation &= (xCalculation >> 31);
                        currentBoundsMaxX = currentBoundsMaxX - xCalculation;
                    }
                    {
                        int yCalculation = (int)point1.y - (int)point2.y;
                        yCalculation &= (yCalculation >> 31);
                        int yMin = (int)point2.y + yCalculation;
                        int yMax = (int)point1.y - yCalculation;

                        yCalculation = currentBoundsMinY - yMin;
                        yCalculation &= (yCalculation >> 31);
                        currentBoundsMinY = yMin + yCalculation;

                        yCalculation = currentBoundsMaxY - yMax;
                        yCalculation &= (yCalculation >> 31);
                        currentBoundsMaxY = currentBoundsMaxY - yCalculation;
                    }
                    {
                        int zCalculation = (int)point1.z - (int)point2.z;
                        zCalculation &= (zCalculation >> 31);
                        int zMin = (int)point2.z + zCalculation;
                        int zMax = (int)point1.z - zCalculation;

                        zCalculation = currentBoundsMinZ - zMin;
                        zCalculation &= (zCalculation >> 31);
                        currentBoundsMinZ = zMin + zCalculation;

                        zCalculation = currentBoundsMaxZ - zMax;
                        zCalculation &= (zCalculation >> 31);
                        currentBoundsMaxZ = currentBoundsMaxZ - zCalculation;
                    }
                }
            }

            private void AddIndices()
            {
                int vertexIndex = vertices.Count;
                indices.Add(vertexIndex++);
                indices.Add(vertexIndex++);
                indices.Add(vertexIndex);
                indices.Add(vertexIndex--);
                indices.Add(vertexIndex);
                indices.Add(vertexIndex += 2);
            }

            private void AppendLineInternal(ref Vector3 start, ref Vector3 end, ref Vector4 dir, ref Vector4 dirPrev1, ref Vector4 dirPrev2,
                Color32 color, float colorIntensity, ref Vector4 fadeLifeTime, float glowWidthModifier, float glowIntensity)
            {
                AddIndices();
                color.a = (byte)Mathf.Lerp(0.0f, 255.0f, colorIntensity * 0.1f);

                Vector4 texCoord = new Vector4(uv1.x, uv1.y, glowWidthModifier, glowIntensity);

                vertices.Add(start);
                lineDirs.Add(dirPrev1);
                colors.Add(color);
                ends.Add(dir);

                vertices.Add(end);
                lineDirs.Add(dir);
                colors.Add(color);
                ends.Add(dir);

                dir.w = -dir.w;

                vertices.Add(start);
                lineDirs.Add(dirPrev2);
                colors.Add(color);
                ends.Add(dir);

                vertices.Add(end);
                lineDirs.Add(dir);
                colors.Add(color);
                ends.Add(dir);

#if UNITY_PRE_5_3

                texCoords.Add(uv1);
				texCoords.Add(uv2);
				texCoords.Add(uv3);
				texCoords.Add(uv4);
				glowModifiers.Add(new Vector2(texCoord.z, texCoord.w));
				glowModifiers.Add(new Vector2(texCoord.z, texCoord.w));
				glowModifiers.Add(new Vector2(texCoord.z, texCoord.w));
				glowModifiers.Add(new Vector2(texCoord.z, texCoord.w));
				fadeXY.Add(new Vector2(fadeLifeTime.x, fadeLifeTime.y));
				fadeXY.Add(new Vector2(fadeLifeTime.x, fadeLifeTime.y));
				fadeXY.Add(new Vector2(fadeLifeTime.x, fadeLifeTime.y));
				fadeXY.Add(new Vector2(fadeLifeTime.x, fadeLifeTime.y));
				fadeZW.Add(new Vector2(fadeLifeTime.z, fadeLifeTime.w));
				fadeZW.Add(new Vector2(fadeLifeTime.z, fadeLifeTime.w));
				fadeZW.Add(new Vector2(fadeLifeTime.z, fadeLifeTime.w));
				fadeZW.Add(new Vector2(fadeLifeTime.z, fadeLifeTime.w));

#else

                texCoordsAndGlowModifiers.Add(texCoord);
                texCoord.x = uv2.x;
                texCoord.y = uv2.y;
                texCoordsAndGlowModifiers.Add(texCoord);
                texCoord.x = uv3.x;
                texCoord.y = uv3.y;
                texCoordsAndGlowModifiers.Add(texCoord);
                texCoord.x = uv4.x;
                texCoord.y = uv4.y;
                texCoordsAndGlowModifiers.Add(texCoord);
                fadeLifetimes.Add(fadeLifeTime);
                fadeLifetimes.Add(fadeLifeTime);
                fadeLifetimes.Add(fadeLifeTime);
                fadeLifetimes.Add(fadeLifeTime);

#endif

                UpdateBounds(ref start, ref end);
            }

            #endregion Private methods
        }

        #endregion LineRendererMesh

        #region Public variables

        /// <summary>
        /// The maximum number of lights to allow for all lightning
        /// </summary>
        public static int MaximumLightCount = 128;

        /// <summary>
        /// The maximum number of lights to create per batch of lightning emitted
        /// </summary>
        public static int MaximumLightsPerBatch = 8;

        /// <summary>
        /// The current minimum delay until anything will start rendering
        /// </summary>
        public float MinimumDelay { get; private set; }

        /// <summary>
        /// Is there any glow for any of the lightning bolts?
        /// </summary>
        public bool HasGlow { get; private set; }

        /// <summary>
        /// Is this lightning bolt active any more?
        /// </summary>
        public bool IsActive { get { return elapsedTime < lifeTime; } }

        /// <summary>
        /// Camera mode
        /// </summary>
        public CameraMode CameraMode { get; private set; }

        public float StartTimeOffset { get; private set; }

        #endregion Public variables

        #region Public methods

        /// <summary>
        /// Default constructor
        /// </summary>
        public LightningBolt()
        {
        }

        public void SetupLightningBolt(LightningBoltDependencies dependencies)
        {
            if (dependencies == null || dependencies.Parameters.Count == 0)
            {
                Debug.LogError("Lightning bolt dependencies must not be null");
                return;
            }
            else if (this.dependencies != null)
            {
                Debug.LogError("This lightning bolt is already in use!");
                return;
            }

            this.dependencies = dependencies;
            CameraMode = dependencies.CameraMode;
            timeSinceLevelLoad = Time.timeSinceLevelLoad;
            CheckForGlow(dependencies.Parameters);
            MinimumDelay = float.MaxValue;
            GetOrCreateLineRenderer();

#if !UNITY_WEBGL

            if (dependencies.ThreadState != null)
            {
                StartTimeOffset = 1.0f / 60.0f;
                dependencies.ThreadState.AddActionForBackgroundThread(ProcessAllLightningParameters);
            }
            else

#endif

            {
                StartTimeOffset = 0.0f;
                ProcessAllLightningParameters();
            }
        }

        public bool Update()
        {
            elapsedTime += Time.deltaTime;
            if (elapsedTime > lifeTime)
            {
                return false;
            }
            else if (hasLight)
            {
                UpdateLights();
            }
            return true;
        }

        public void Cleanup()
        {
            foreach (LightningBoltSegmentGroup g in segmentGroupsWithLight)
            {
                // cleanup lights
                foreach (Light l in g.Lights)
                {
                    CleanupLight(l);
                }
                g.Lights.Clear();
            }
            foreach (LightningBoltSegmentGroup g in segmentGroups)
            {
                groupCache.Add(g);
            }
            hasLight = false;
            elapsedTime = 0.0f;
            lifeTime = 0.0f;
            if (dependencies != null)
            {
                dependencies.ReturnToCache(dependencies);
                dependencies = null;
            }

            // return all line renderers to cache
            foreach (LineRendererMesh m in activeLineRenderers)
            {
                if (m != null)
                {
                    m.Reset();
                    lineRendererCache.Add(m);
                }
            }
            segmentGroups.Clear();
            segmentGroupsWithLight.Clear();
            activeLineRenderers.Clear();
        }

        public LightningBoltSegmentGroup AddGroup()
        {
            LightningBoltSegmentGroup group;
            if (groupCache.Count == 0)
            {
                group = new LightningBoltSegmentGroup();
            }
            else
            {
                int index = groupCache.Count - 1;
                group = groupCache[index];
                group.Reset();
                groupCache.RemoveAt(index);
            }
            segmentGroups.Add(group);
            return group;
        }

        /// <summary>
        /// Clear out all cached objects to free up memory
        /// </summary>
        public static void ClearCache()
        {
            foreach (LineRendererMesh obj in lineRendererCache)
            {
                if (obj != null)
                {
                    GameObject.Destroy(obj.GameObject);
                }
            }
            foreach (Light obj in lightCache)
            {
                if (obj != null)
                {
                    GameObject.Destroy(obj.gameObject);
                }
            }
            lineRendererCache.Clear();
            lightCache.Clear();
            groupCache.Clear();
        }

        #endregion Public methods

        #region Private variables

        // required dependencies to create lightning bolts
        private LightningBoltDependencies dependencies;

        // how long this bolt has been alive
        private float elapsedTime;

        // total life span of this bolt
        private float lifeTime;

        // does this lightning bolt have light?
        private bool hasLight;

        // saved in case of threading
        private float timeSinceLevelLoad;

        private readonly List<LightningBoltSegmentGroup> segmentGroups = new List<LightningBoltSegmentGroup>();
        private readonly List<LightningBoltSegmentGroup> segmentGroupsWithLight = new List<LightningBoltSegmentGroup>();
        private readonly List<LineRendererMesh> activeLineRenderers = new List<LineRendererMesh>();

        private static int lightCount;
        private static readonly List<LineRendererMesh> lineRendererCache = new List<LineRendererMesh>();
        private static readonly List<LightningBoltSegmentGroup> groupCache = new List<LightningBoltSegmentGroup>();
        private static readonly List<Light> lightCache = new List<Light>();

        #endregion Private variables

        #region Private methods

        private void CleanupLight(Light l)
        {
            if (l != null)
            {
                dependencies.LightRemoved(l);
                lightCache.Add(l);
                l.gameObject.SetActive(false);
                lightCount--;
            }
        }

        private void EnableLineRenderer(LineRendererMesh lineRenderer, int tag)
        {
            bool shouldPopulate = (lineRenderer != null && lineRenderer.GameObject != null && lineRenderer.Tag == tag && IsActive);
            if (shouldPopulate)
            {
                lineRenderer.PopulateMesh();
            }
        }

        private IEnumerator EnableLastRendererCoRoutine()
        {
            LineRendererMesh lineRenderer = activeLineRenderers[activeLineRenderers.Count - 1];
            int tag = ++lineRenderer.Tag; // in case it gets cleaned up for later

            yield return new WaitForSeconds(MinimumDelay);

            EnableLineRenderer(lineRenderer, tag);
        }

        private LineRendererMesh GetOrCreateLineRenderer()
        {
            LineRendererMesh lineRenderer;

			while (true)
            {
                if (lineRendererCache.Count == 0)
                {
                    lineRenderer = new LineRendererMesh();
                }
                else
                {
                    int index = lineRendererCache.Count - 1;
                    lineRenderer = lineRendererCache[index];
                    if (lineRenderer == null)
                    {
                        // destroyed by some other means, try again for cache...
                        continue;
                    }
                    lineRendererCache.RemoveAt(index);
                }
                break;
            }

            // clear parent - this ensures that the rotation and scale can be reset before assigning a new parent
            lineRenderer.GameObject.transform.parent = null;
            lineRenderer.GameObject.transform.rotation = Quaternion.identity;
            lineRenderer.GameObject.transform.localScale = Vector3.one;
            lineRenderer.GameObject.transform.parent = dependencies.Parent.transform;
            lineRenderer.GameObject.layer = dependencies.Parent.layer; // maintain the layer of the parent

            if (dependencies.UseWorldSpace)
            {
                lineRenderer.GameObject.transform.position = Vector3.zero;
            }
            else
            {
                lineRenderer.GameObject.transform.localPosition = Vector3.zero;
            }

            lineRenderer.Material = (HasGlow ? dependencies.LightningMaterialMesh : dependencies.LightningMaterialMeshNoGlow);
            lineRenderer.MeshRenderer.sortingLayerName = dependencies.SortLayerName;
            lineRenderer.MeshRenderer.sortingOrder = dependencies.SortOrderInLayer;

            activeLineRenderers.Add(lineRenderer);

            return lineRenderer;
        }

        private void RenderGroup(LightningBoltSegmentGroup group, LightningBoltParameters p)
        {
            if (group.SegmentCount == 0)
            {
                return;
            }

            LineRendererMesh currentLineRenderer = activeLineRenderers[activeLineRenderers.Count - 1];
            float timeStart = timeSinceLevelLoad + group.Delay + StartTimeOffset;
            Vector4 fadeLifeTime = new Vector4(timeStart, timeStart + group.PeakStart, timeStart + group.PeakEnd, timeStart + group.LifeTime);
            float radius = group.LineWidth * 0.5f * LightningBoltParameters.Scale;
            int lineCount = (group.Segments.Count - group.StartIndex);
            float radiusStep = (radius - (radius * group.EndWidthMultiplier)) / (float)lineCount;

            // growth multiplier
            float timeStep, timeOffset;
            if (p.GrowthMultiplier > 0.0f)
            {
                timeStep = (group.LifeTime / (float)lineCount) * p.GrowthMultiplier;
                timeOffset = 0.0f;
            }
            else
            {
                timeStep = 0.0f;
                timeOffset = 0.0f;
            }

            // if we have filled up the mesh, we need to start a new line renderer
            if (!currentLineRenderer.PrepareForLines(lineCount))
            {

#if !UNITY_WEBGL

                if (dependencies.ThreadState != null)
                {
                    // we need to block until this action is run, Unity objects can only be modified and created on the main thread
                    dependencies.ThreadState.AddActionForMainThread(() =>
                    {
                        EnableCurrentLineRenderer();
                        currentLineRenderer = GetOrCreateLineRenderer();
                    }, true);
                }
                else

#endif

                {
                    EnableCurrentLineRenderer();
                    currentLineRenderer = GetOrCreateLineRenderer();
                }
            }

            currentLineRenderer.BeginLine(group.Segments[group.StartIndex].Start, group.Segments[group.StartIndex].End, radius, group.Color, p.Intensity, fadeLifeTime, p.GlowWidthMultiplier, p.GlowIntensity);
            for (int i = group.StartIndex + 1; i < group.Segments.Count; i++)
            {
                radius -= radiusStep;
                if (p.GrowthMultiplier < 1.0f)
                {
                    timeOffset += timeStep;
                    fadeLifeTime = new Color(timeStart + timeOffset, timeStart + group.PeakStart + timeOffset, timeStart + group.PeakEnd, timeStart + group.LifeTime);
                }
                currentLineRenderer.AppendLine(group.Segments[i].Start, group.Segments[i].End, radius, group.Color, p.Intensity, fadeLifeTime, p.GlowWidthMultiplier, p.GlowIntensity);
            }
        }

        private static IEnumerator NotifyBolt(LightningBoltDependencies dependencies, LightningBoltParameters p, Vector3 start, Vector3 end)
        {
            float delay = p.delaySeconds;
            float lifeTime = p.LifeTime;
            yield return new WaitForSeconds(delay);
            if (dependencies.LightningBoltStarted != null)
            {
                dependencies.LightningBoltStarted(p, start, end);
            }
            yield return new WaitForSeconds(lifeTime);
            if (dependencies.LightningBoltEnded != null)
            {
                dependencies.LightningBoltEnded(p, start, end);
            }
            LightningBoltParameters.ReturnParametersToCache(p);
        }

        private void ProcessParameters(LightningBoltParameters p, RangeOfFloats delay)
        {
            Vector3 start, end;
            MinimumDelay = Mathf.Min(delay.Minimum, MinimumDelay);
            p.delaySeconds = delay.Random(p.Random);
            p.generationWhereForksStop = p.Generations - p.GenerationWhereForksStopSubtractor;
            lifeTime = Mathf.Max(p.LifeTime + p.delaySeconds, lifeTime);
            p.forkednessCalculated = (int)Mathf.Ceil(p.Forkedness * (float)p.Generations);
            if (p.Generations > 0)
            {
                LightningBoltDependencies dependenciesRef = dependencies;
                p.Generator = p.Generator ?? LightningGenerator.GeneratorInstance;
                p.Generator.GenerateLightningBolt(this, p, out start, out end);

#if !UNITY_WEBGL

                if (dependenciesRef.ThreadState != null)
                {
                    dependenciesRef.ThreadState.AddActionForMainThread(() =>
                    {
                        dependenciesRef.StartCoroutine(NotifyBolt(dependenciesRef, p, start, end));
                    });
                }
                else

#endif

                {
                    dependenciesRef.StartCoroutine(NotifyBolt(dependenciesRef, p, start, end));
                }
            }
        }

        private void ProcessAllLightningParameters()
        {
            int maxLightsForEachParameters = MaximumLightsPerBatch / dependencies.Parameters.Count;
            RangeOfFloats delay = new RangeOfFloats();
            List<int> groupIndexes = new List<int>(dependencies.Parameters.Count + 1);
            int i = 0;

#if ENABLE_PROFILING

            System.Diagnostics.Stopwatch w = System.Diagnostics.Stopwatch.StartNew();

#endif

            foreach (LightningBoltParameters p in dependencies.Parameters)
            {
                delay.Minimum = p.DelayRange.Minimum + p.Delay;
                delay.Maximum = p.DelayRange.Maximum + p.Delay;
                p.maxLights = maxLightsForEachParameters;
                groupIndexes.Add(segmentGroups.Count);
                ProcessParameters(p, delay);
            }
            groupIndexes.Add(segmentGroups.Count);

#if ENABLE_PROFILING

            w.Stop();
            UnityEngine.Debug.LogFormat("GENERATE: {0}", w.Elapsed.TotalMilliseconds);
            w.Reset();
            w.Start();

#endif

            foreach (LightningBoltParameters p in dependencies.Parameters)
            {
                RenderLightningBolt(p.quality, p.Generations, groupIndexes[i], groupIndexes[++i], p);
            }

#if ENABLE_PROFILING

            w.Stop();
            UnityEngine.Debug.LogFormat("RENDER: {0}", w.Elapsed.TotalMilliseconds);

#endif

#if !UNITY_WEBGL

            if (dependencies.ThreadState != null)
            {
                dependencies.ThreadState.AddActionForMainThread(EnableCurrentLineRendererFromThread);
            }
            else

#endif

            {
                EnableCurrentLineRenderer();
                dependencies.AddActiveBolt(this);
            }
        }

        private void EnableCurrentLineRendererFromThread()
        {
            EnableCurrentLineRenderer();

#if !UNITY_WEBGL

            // clear the thread state, we verify in the Cleanup method that this is nulled out to ensure we are not cleaning up lightning that is still being generated
            dependencies.ThreadState = null;

#endif

            dependencies.AddActiveBolt(this);
        }

        private void EnableCurrentLineRenderer()
        {
            // make sure the last renderer gets enabled at the appropriate time
            if (MinimumDelay <= 0.0f)
            {
                EnableLineRenderer(activeLineRenderers[activeLineRenderers.Count - 1], activeLineRenderers[activeLineRenderers.Count - 1].Tag);
            }
            else
            {
                dependencies.StartCoroutine(EnableLastRendererCoRoutine());
            }
        }

        private void RenderParticleSystems(Vector3 start, Vector3 end, float trunkWidth, float lifeTime, float delaySeconds)
        {
            // only emit particle systems if we have a trunk - example, cloud lightning should not emit particles
            if (trunkWidth > 0.0f)
            {
                if (dependencies.OriginParticleSystem != null)
                {
                    // we have a strike, create a particle where the lightning is coming from
                    dependencies.StartCoroutine(GenerateParticleCoRoutine(dependencies.OriginParticleSystem, start, delaySeconds));
                }
                if (dependencies.DestParticleSystem != null)
                {
                    dependencies.StartCoroutine(GenerateParticleCoRoutine(dependencies.DestParticleSystem, end, delaySeconds + (lifeTime * 0.8f)));
                }
            }
        }

        private void RenderLightningBolt(LightningBoltQualitySetting quality, int generations, int startGroupIndex, int endGroupIndex, LightningBoltParameters parameters)
        {
            if (segmentGroups.Count == 0 || startGroupIndex >= segmentGroups.Count || endGroupIndex > segmentGroups.Count)
            {
                return;
            }

            LightningLightParameters lp = parameters.LightParameters;
            if (lp != null)
            {
                if ((hasLight |= lp.HasLight))
                {
                    lp.LightPercent = Mathf.Clamp(lp.LightPercent, Mathf.Epsilon, 1.0f);
                    lp.LightShadowPercent = Mathf.Clamp(lp.LightShadowPercent, 0.0f, 1.0f);
                }
                else
                {
                    lp = null;
                }
            }

            LightningBoltSegmentGroup mainTrunkGroup = segmentGroups[startGroupIndex];
            Vector3 start = mainTrunkGroup.Segments[mainTrunkGroup.StartIndex].Start;
            Vector3 end = mainTrunkGroup.Segments[mainTrunkGroup.StartIndex + mainTrunkGroup.SegmentCount - 1].End;
            parameters.FadePercent = Mathf.Clamp(parameters.FadePercent, 0.0f, 0.5f);

            for (int i = startGroupIndex; i < endGroupIndex; i++)
            {
                LightningBoltSegmentGroup group = segmentGroups[i];
                group.Delay = parameters.delaySeconds;
                group.LifeTime = parameters.LifeTime;
                group.PeakStart = group.LifeTime * parameters.FadePercent;
                group.PeakEnd = group.LifeTime - group.PeakStart;
                group.LightParameters = lp;
                RenderGroup(group, parameters);
            }

#if !UNITY_WEBGL

            if (dependencies.ThreadState != null)
            {
                dependencies.ThreadState.AddActionForMainThread(() =>
                {
                    RenderParticleSystems(start, end, parameters.TrunkWidth, parameters.LifeTime, parameters.delaySeconds);

                    // create lights only on the main trunk
                    if (lp != null)
                    {
                        CreateLightsForGroup(segmentGroups[startGroupIndex], lp, quality, parameters.maxLights);
                    }
                });
            }
            else

#endif

            {
                RenderParticleSystems(start, end, parameters.TrunkWidth, parameters.LifeTime, parameters.delaySeconds);

                // create lights only on the main trunk
                if (lp != null)
                {
                    CreateLightsForGroup(segmentGroups[startGroupIndex], lp, quality, parameters.maxLights);
                }
            }
        }

        private void CreateLightsForGroup(LightningBoltSegmentGroup group, LightningLightParameters lp, LightningBoltQualitySetting quality, int maxLights)
        {
            if (lightCount == MaximumLightCount || maxLights <= 0)
            {
                return;
            }

            segmentGroupsWithLight.Add(group);

            int segmentCount = group.SegmentCount;
            float lightPercent, lightShadowPercent;
            if (quality == LightningBoltQualitySetting.LimitToQualitySetting)
            {
                int level = QualitySettings.GetQualityLevel();
                LightningQualityMaximum maximum;
                if (LightningBoltParameters.QualityMaximums.TryGetValue(level, out maximum))
                {
                    lightPercent = Mathf.Min(lp.LightPercent, maximum.MaximumLightPercent);
                    lightShadowPercent = Mathf.Min(lp.LightShadowPercent, maximum.MaximumShadowPercent);
                }
                else
                {
                    Debug.LogError("Unable to read lightning quality for level " + level.ToString());
                    lightPercent = lp.LightPercent;
                    lightShadowPercent = lp.LightShadowPercent;
                }
            }
            else
            {
                lightPercent = lp.LightPercent;
                lightShadowPercent = lp.LightShadowPercent;
            }

            maxLights = Mathf.Max(1, Mathf.Min(maxLights, (int)(segmentCount * lightPercent)));
            int nthLight = Mathf.Max(1, (int)((segmentCount / maxLights)));
            int nthShadows = maxLights - (int)((float)maxLights * lightShadowPercent);

            int nthShadowCounter = nthShadows;

            // add lights evenly spaced
            for (int i = group.StartIndex + (int)(nthLight * 0.5f); i < group.Segments.Count; i += nthLight)
            {
                if (AddLightToGroup(group, lp, i, nthLight, nthShadows, ref maxLights, ref nthShadowCounter))
                {
                    return;
                }
            }

            // Debug.Log("Lightning light count: " + lightCount.ToString());
        }

        private bool AddLightToGroup(LightningBoltSegmentGroup group, LightningLightParameters lp, int segmentIndex,
            int nthLight, int nthShadows, ref int maxLights, ref int nthShadowCounter)
        {
            Light light = GetOrCreateLight(lp);
            group.Lights.Add(light);
            Vector3 pos = (group.Segments[segmentIndex].Start + group.Segments[segmentIndex].End) * 0.5f;
            if (dependencies.CameraIsOrthographic)
            {
                pos.z = dependencies.CameraPos.z;
            }
            light.gameObject.transform.position = pos;
            if (lp.LightShadowPercent == 0.0f || ++nthShadowCounter < nthShadows)
            {
                light.shadows = LightShadows.None;
            }
            else
            {
                light.shadows = LightShadows.Soft;
                nthShadowCounter = 0;
            }

            // return true if no more lights possible, false otherwise
            return (++lightCount == MaximumLightCount || --maxLights == 0);
        }

        private Light GetOrCreateLight(LightningLightParameters lp)
        {
            Light light;
            while (true)
            {
                if (lightCache.Count == 0)
                {
                    GameObject lightningLightObject = new GameObject("LightningBoltLight");

#if UNITY_EDITOR

                    lightningLightObject.hideFlags = HideFlags.HideInInspector | HideFlags.HideInHierarchy;

#endif

                    light = lightningLightObject.AddComponent<Light>();
                    light.type = LightType.Point;
                    break;
                }
                else
                {
                    light = lightCache[lightCache.Count - 1];
                    lightCache.RemoveAt(lightCache.Count - 1);
                    if (light == null)
                    {
                        // may have been disposed or the level re-loaded
                        continue;
                    }
                    break;
                }
            }

#if UNITY_4

#else

            light.bounceIntensity = lp.BounceIntensity;
            light.shadowNormalBias = lp.ShadowNormalBias;

#endif

            light.color = lp.LightColor;
            light.renderMode = lp.RenderMode;
            light.range = lp.LightRange;
            light.shadowStrength = lp.ShadowStrength;
            light.shadowBias = lp.ShadowBias;
            light.intensity = 0.0f;
            light.gameObject.transform.parent = dependencies.Parent.transform;
            light.gameObject.SetActive(true);

            dependencies.LightAdded(light);

            return light;
        }

        private void UpdateLight(LightningLightParameters lp, IEnumerable<Light> lights, float delay, float peakStart, float peakEnd, float lifeTime)
        {
            if (elapsedTime < delay)
            {
                return;
            }

            // depending on whether we have hit the mid point of our lifetime, fade the light in or out
            float realElapsedTime = elapsedTime - delay;
            if (realElapsedTime >= peakStart)
            {
                if (realElapsedTime <= peakEnd)
                {
                    // fully lit
                    foreach (Light l in lights)
                    {
                        l.intensity = lp.LightIntensity;
                    }
                }
                else
                {
                    // fading out
                    float lerp = (realElapsedTime - peakEnd) / (lifeTime - peakEnd);
                    foreach (Light l in lights)
                    {
                        l.intensity = Mathf.Lerp(lp.LightIntensity, 0.0f, lerp);
                    }
                }
            }
            else
            {
                // fading in
                float lerp = realElapsedTime / peakStart;
                foreach (Light l in lights)
                {
                    l.intensity = Mathf.Lerp(0.0f, lp.LightIntensity, lerp);
                }
            }
        }

        private void UpdateLights()
        {
            foreach (LightningBoltSegmentGroup group in segmentGroupsWithLight)
            {
                UpdateLight(group.LightParameters, group.Lights, group.Delay, group.PeakStart, group.PeakEnd, group.LifeTime);
            }
        }

        private IEnumerator GenerateParticleCoRoutine(ParticleSystem p, Vector3 pos, float delay)
        {
            yield return new WaitForSeconds(delay);

            p.transform.position = pos;

#if UNITY_PRE_5_3

            p.Emit((int)p.emissionRate);

#else

            int count;
            if (p.emission.burstCount > 0)
            {
                ParticleSystem.Burst[] bursts = new ParticleSystem.Burst[p.emission.burstCount];
                p.emission.GetBursts(bursts);
                count = UnityEngine.Random.Range(bursts[0].minCount, bursts[0].maxCount + 1);
                p.Emit(count);
            }
            else
            {

#if UNITY_5_6_OR_NEWER

                ParticleSystem.MinMaxCurve rate = p.emission.rateOverTime;

#else

                ParticleSystem.MinMaxCurve rate = p.emission.rate;

#endif

                count = (int)((rate.constantMax - rate.constantMin) * 0.5f);
                count = UnityEngine.Random.Range(count, count * 2);
                p.Emit(count);
            }

#endif

        }

        private void CheckForGlow(IEnumerable<LightningBoltParameters> parameters)
        {
            // we need to know if there is glow so we can choose the glow or non-glow setting in the renderer
            foreach (LightningBoltParameters p in parameters)
            {
                HasGlow = (p.GlowIntensity >= Mathf.Epsilon && p.GlowWidthMultiplier >= Mathf.Epsilon);

                if (HasGlow)
                {
                    break;
                }
            }
        }

        #endregion Private methods
    }

#if !UNITY_WEBGL

    /// <summary>
    /// Lightning threading state
    /// </summary>
    public class LightningThreadState
    {

#if TASK_AVAILABLE

        private Task lightningThread;

#else

        /// <summary>
        /// Lightning thread
        /// </summary>
        private Thread lightningThread;

#endif

        /// <summary>
        /// Lightning thread event to notify background action available
        /// </summary>
        private AutoResetEvent lightningThreadEvent = new AutoResetEvent(false);

        /// <summary>
        /// List of background actions
        /// </summary>
        private readonly Queue<System.Action> actionsForBackgroundThread = new Queue<System.Action>();

        /// <summary>
        /// List of main thread actions and optional events to signal
        /// </summary>
        private readonly Queue<KeyValuePair<System.Action, ManualResetEvent>> actionsForMainThread = new Queue<KeyValuePair<System.Action, ManualResetEvent>>();

        /// <summary>
        /// Set to false to terminate
        /// </summary>
        public bool Running = true;

        private void BackgroundThreadMethod()
        {
            System.Action action = null;
            while (Running)
            {
                try
                {
                    if (!lightningThreadEvent.WaitOne(500))
                    {
                        continue;
                    }

                    tryActionAgain:
                    lock (actionsForBackgroundThread)
                    {
                        if (actionsForBackgroundThread.Count == 0)
                        {
                            continue;
                        }
                        action = actionsForBackgroundThread.Dequeue();
                    }
                    action();
                    goto tryActionAgain;
                }
                catch (Exception ex)
                {
                    Debug.LogErrorFormat("Lightning thread exception: {0}", ex);
                }
            }
        }

        /// <summary>
        /// Constructor - starts the thread
        /// </summary>
        public LightningThreadState()
        {

#if TASK_AVAILABLE

            lightningThread = Task.Factory.StartNew(BackgroundThreadMethod);

#else

            lightningThread = new Thread(new ThreadStart(BackgroundThreadMethod))
            {
                IsBackground = true,
                Name = "LightningBoltScriptThread"
            };
            lightningThread.Start();

#endif

        }

        /// <summary>
        /// Execute any main thread actions from the main thread
        /// </summary>
        public void UpdateMainThreadActions()
        {
            KeyValuePair<System.Action, ManualResetEvent> kv;

            mainThreadLoop:
            {
                lock (actionsForMainThread)
                {
                    if (actionsForMainThread.Count == 0)
                    {
                        return;
                    }
                    kv = actionsForMainThread.Dequeue();
                }
                kv.Key();
                if (kv.Value != null)
                {
                    kv.Value.Set();
                }
                goto mainThreadLoop;
            }
        }

        /// <summary>
        /// Add a main thread action
        /// </summary>
        /// <param name="a">Action</param>
        /// <param name="waitForAction">True to wait for completion, false if not</param>
        public void AddActionForMainThread(System.Action a, bool waitForAction = false)
        {
            ManualResetEvent evt = (waitForAction ? new ManualResetEvent(false) : null);
            lock (actionsForMainThread)
            {
                actionsForMainThread.Enqueue(new KeyValuePair<System.Action, ManualResetEvent>(a, evt));
            }
            if (evt != null)
            {
                evt.WaitOne(10000);
            }
        }

        /// <summary>
        /// Add a background thread action
        /// </summary>
        /// <param name="a">Action</param>
        public void AddActionForBackgroundThread(System.Action a)
        {
            lock (actionsForBackgroundThread)
            {
                actionsForBackgroundThread.Enqueue(a);
            }
            lightningThreadEvent.Set();
        }
    }

#endif

}
