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
using System.IO;
using System.Reflection;

using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    /// <summary>
    /// Interface for a weather maker property transition
    /// </summary>
    [Serializable]
    public class WeatherMakerPropertyTransition : ISerializationCallbackReceiver
    {
        [SerializeField]
        private byte[] valueBytes;
        [NonSerialized]
        [Tooltip("The value to transition to")]
        public object Value;

        [SerializeField]
        [Tooltip("The animation curve")]
        public AnimationCurve Curve;

        [SerializeField]
        [Tooltip("The target to set the value on")]
        public MonoBehaviour Target;

        [SerializeField]
        [Tooltip("The field name to set")]
        public string FieldName;

#if UNITY_EDITOR

        /// <summary>
        /// Editor only
        /// </summary>
        public PopupList FieldNamePopup { get; set; }

#endif

        /// <summary>
        /// Get the start value
        /// </summary>
        public object StartValue { get; private set; }

        /// <summary>
        /// Percent elapsed for this transition (0 to 1)
        /// </summary>
        public float PercentElapsed { get; set; }

        private System.Reflection.FieldInfo fieldInfo;

        /// <summary>
        /// Update
        /// </summary>
        public void Update(float percentElapsed)
        {
            float curve = Curve.Evaluate(percentElapsed);
            if (fieldInfo.FieldType == typeof(float))
            {
                float val = Mathf.Lerp((float)StartValue, (float)Value, curve);
                fieldInfo.SetValue(Target, val);
            }
            else if (fieldInfo.FieldType == typeof(Color))
            {
                Color val = Color.Lerp((Color)StartValue, (Color)Value, curve);
                fieldInfo.SetValue(Target, val);
            }
            else if (fieldInfo.FieldType == typeof(Vector2))
            {
                Vector2 val = Vector2.Lerp((Vector2)StartValue, (Vector2)Value, curve);
                fieldInfo.SetValue(Target, val);
            }
            else if (fieldInfo.FieldType == typeof(Vector3))
            {
                Vector2 val = Vector2.Lerp((Vector3)StartValue, (Vector3)Value, curve);
                fieldInfo.SetValue(Target, val);
            }
            else if (fieldInfo.FieldType == typeof(Vector4))
            {
                Vector2 val = Vector2.Lerp((Vector4)StartValue, (Vector4)Value, curve);
                fieldInfo.SetValue(Target, val);
            }
            else if (curve >= 0.5f)
            {
                // no lerp possible, just set the value if the curve is over the half way mark
                fieldInfo.SetValue(Target, Value);
            }
        }

        public void UpdateStartValue()
        {
            if (fieldInfo == null)
            {

#if NETFX_CORE

                fieldInfo = Target.GetType().GetRuntimeField(FieldName);

#else

                fieldInfo = Target.GetType().GetField(FieldName, BindingFlags.GetField | BindingFlags.SetField | BindingFlags.Instance | BindingFlags.Public);

#endif

            }

            StartValue = fieldInfo.GetValue(Target);
        }

        public void OnBeforeSerialize()
        {
            valueBytes = SerializationHelper.Serialize(Value);
        }

        public void OnAfterDeserialize()
        {
            Type t = null;
            if (Target != null)
            {

#if NETFX_CORE

                FieldInfo field = Target.GetType().GetRuntimeField(FieldName);

#else

                FieldInfo field = Target.GetType().GetField(FieldName, BindingFlags.GetField | BindingFlags.SetField | BindingFlags.Instance | BindingFlags.Public);

#endif

                if (field != null)
                {
                    t = field.FieldType;
                }
            }
            Value = SerializationHelper.Deserialize(valueBytes, t);
        }
    }

    [Serializable]
    public class WeatherMakerPropertyTransitionGroup
    {
        [SerializeField]
        public string Title;

        [SerializeField]
        public List<WeatherMakerPropertyTransition> Transitions;
    }

    [Serializable]
    public class WeatherMakerTransitionGroupWeight
    {
        [SerializeField]
        [Range(0.0f, 100.0f)]
        [Tooltip("Transition weight")]
        public int Weight;

        [SerializeField]
        [Tooltip("Transition group")]
        public WeatherMakerPropertyTransitionGroup TransitionGroup;

        [NonSerialized]
        [HideInInspector]
        public readonly HashSet<string> FieldNames = new HashSet<string>();

        [NonSerialized]
        [HideInInspector]
        public System.Action<float> CustomUpdater;

#if UNITY_EDITOR

        /// <summary>
        /// Editor only
        /// </summary>
        public UnityEditorInternal.ReorderableList TransitionList { get; set; }

#endif

        /// <summary>
        /// Update all transitions
        /// </summary>
        /// <param name="elapsedPercent">Percent elapsed</param>
        public void Update(float elapsedPercent)
        {
            foreach (WeatherMakerPropertyTransition t in TransitionGroup.Transitions)
            {
                t.Update(elapsedPercent);
            }
            if (CustomUpdater != null)
            {
                CustomUpdater(elapsedPercent);
            }
        }

        public override string ToString()
        {
            return TransitionGroup.Title + ", transition count: " + TransitionGroup.Transitions.Count;
        }
    }

    /// <summary>
    /// A script that allows automatic weather and sky conditions
    /// </summary>
    public class WeatherMakerWeatherManagerScript : MonoBehaviour
    {
        /// <summary>
        /// Rendered by WeatherMakerWeatherManagerEditor
        /// </summary>
        [HideInInspector]
        [Tooltip("A list of possible transitions with weights. Higher weighted transitions are more likely to take place.")]
        public List<WeatherMakerTransitionGroupWeight> Transitions;

        [SingleLineClamp("Random range for transition duration in game time (Time.deltaTime)", 1.0f, 1000.0f)]
        public RangeOfFloats TransitionTime = new RangeOfFloats { Minimum = 30.0f, Maximum = 120.0f };

        [SingleLineClamp("Random range for transition hold duration in game time (Time.deltaTime) - this is how long the transition stays at peak before a new transition begins.", 1.0f, 1000.0f)]
        public RangeOfFloats HoldTime = new RangeOfFloats { Minimum = 60.0f, Maximum = 300.0f };

        [Tooltip("Whether to use the built in cloud animation. Default is true. Set to false if you are animating things like cloud cover etc., instead of just setting the cloud type.")]
        public bool UseBuiltInCloudAnimation = true;

        private WeatherMakerTransitionGroupWeight currentTransition;
        private int maxWeight;
        private float elapsedTime;
        private float totalTime;
        private float transitionTime; // how long the transition lasts
        private float holdTime; // how long transition lasts after it is complete before a new transition is chosen
        private bool holdingTransition;
        private System.Random random;
        private int randomSeed = System.Environment.TickCount;
        private float startPrecipitationIntensity;
        private float startFogDensity;
        private float startWindIntensity;
        private bool initialized;

        /// <summary>
        /// Begin a new transition group
        /// </summary>
        /// <param name="index">Index of the group in Transitions</param>
        /// <param name="randomSeed">Random seed</param>
        public void StartNewTransitionGroup(int index, int randomSeed)
        {
            Initialize(randomSeed);
            currentTransition = Transitions[index];
            transitionTime = TransitionTime.Random(random);
            holdTime = HoldTime.Random(random);
            totalTime = transitionTime;

            Debug.LogFormat("Starting Weather Transition {0}, transition time: {1:0.00}, hold time: {2:0.00}", currentTransition.TransitionGroup.Title, transitionTime, holdTime);

            // turn off things that don't need to be setup on every transition
            WeatherMakerScript.Instance.LightningScript.EnableLightning = false;
            foreach (WeatherMakerPropertyTransition t in currentTransition.TransitionGroup.Transitions)
            {
                t.UpdateStartValue();
            }

            // store properties in case they are not animated - for this we will lerp these to low or 0 values as we want them essentially off by default unless they are animated
            startPrecipitationIntensity = WeatherMakerScript.Instance.PrecipitationIntensity;
            startFogDensity = WeatherMakerScript.Instance.FogScript.FogDensity;
            startWindIntensity = WeatherMakerScript.Instance.WindIntensity;

            currentTransition.CustomUpdater = CustomUpdater;

            // how long the group transition lasts
            WeatherMakerScript.Instance.CloudChangeDuration = (UseBuiltInCloudAnimation ? transitionTime : 0.0f);
            holdingTransition = false;

            WeatherMakerScript.Instance.RaiseWeatherManagerTransitionStarted(WeatherMakerScript.Instance.WeatherManagers.IndexOf(this), index, randomSeed);
        }

        private void Initialize(int randomSeed)
        {
            // always ensure we have a random that matches the seed, this allows clients to get same weather pattern as server
            if (random == null || randomSeed != this.randomSeed)
            {
                random = new System.Random(this.randomSeed = randomSeed);
            }

            if (initialized)
            {
                return;
            }
            initialized = true;
            Transitions.Sort((t1, t2) =>
            {
                return t1.Weight.CompareTo(t2.Weight);
            });
            foreach (WeatherMakerTransitionGroupWeight w in Transitions)
            {
                maxWeight += w.Weight;
                foreach (WeatherMakerPropertyTransition g in w.TransitionGroup.Transitions)
                {
                    w.FieldNames.Add(g.FieldName);
                }
            }
            maxWeight++;
            WeatherMakerScript.Instance.PrecipitationChangeDelay = 0.0f;
            WeatherMakerScript.Instance.PrecipitationChangeDuration = 0.0f; // we are animating the precipitation intensity in this script, do not use the built in animations
            WeatherMakerScript.Instance.PrecipitationIntensity = 0.0f; // start off with no precipitation intensity to correctly animate the first precipitation
        }

        private void CustomUpdater(float percent)
        {
            if (!currentTransition.FieldNames.Contains("PrecipitationIntensity"))
            {
                // get rid of precipitation
                WeatherMakerScript.Instance.PrecipitationIntensity = Mathf.Lerp(startPrecipitationIntensity, 0.0f, Mathf.Min(1.0f, percent * 4.0f));
            }
            if (!currentTransition.FieldNames.Contains("FogDensity"))
            {
                // get rid of fog
                WeatherMakerScript.Instance.FogScript.FogDensity = Mathf.Lerp(startFogDensity, 0.0f, Mathf.Min(1.0f, percent * 2.0f));
            }
            if (!currentTransition.FieldNames.Contains("WindIntensity"))
            {
                // reduce wind to low
                WeatherMakerScript.Instance.WindIntensity = Mathf.Lerp(startWindIntensity, 0.02f, Mathf.Min(1.0f, percent * 8.0f));
            }
        }

        /// <summary>
        /// Valid field types to transition with
        /// </summary>
        public static readonly List<Type> ValidFieldTypes = new List<Type>
        {
            typeof(Enum),
            typeof(float),
            typeof(int),
            typeof(bool),
            typeof(Color),
            typeof(Vector2),
            typeof(Vector3),
            typeof(Vector4),
            typeof(RangeOfFloats),
            typeof(RangeOfIntegers)
        };

        private void PickTransition()
        {
            Initialize(randomSeed);

            // clients will get a transition sent to them
            if (!WeatherMakerScript.Instance.NetworkScript.IsServer)
            {
                return;
            }

            int weight = UnityEngine.Random.Range(1, maxWeight + 1);
            int sum = 0;
            WeatherMakerTransitionGroupWeight w;

            for (int i = 0; i < Transitions.Count; i++)
            {
                w = Transitions[i];
                sum += w.Weight;
                if (sum >= weight)
                {
                    StartNewTransitionGroup(i, randomSeed);
                    break;
                }
            }
        }

        private void OnDisable()
        {
            elapsedTime = totalTime = holdTime = 0.0f;
            holdingTransition = false;
            currentTransition = null;
        }

        private void Update()
        {
            elapsedTime += Time.deltaTime;
            if (elapsedTime >= totalTime)
            {
                elapsedTime = 0.0f;
                if (holdingTransition || currentTransition == null)
                {
                    currentTransition = null;
                    PickTransition();
                }
                else
                {
                    // finish the transition
                    currentTransition.Update(1.0f);
                    holdingTransition = true;
                    totalTime = holdTime;
                }
            }
            else if (currentTransition != null && !holdingTransition)
            {
                currentTransition.Update(elapsedTime / totalTime);
            }
        }
    }
}
