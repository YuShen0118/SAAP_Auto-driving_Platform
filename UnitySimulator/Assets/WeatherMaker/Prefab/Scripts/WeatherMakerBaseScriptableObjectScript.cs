using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerBaseScriptableObjectScript : ScriptableObject
    {
        private bool wasDisabled;

        [Tooltip("Whether this object is disabled")]
        public bool Disabled;

        public virtual void Awake()
        {
        }

        public virtual void Update()
        {
        }

        protected virtual void OnDestroy()
        {
            
        }

        protected virtual void OnDisable()
        {
            wasDisabled = true;
            Disabled = true;
        }

        protected virtual void OnEnable()
        {
            if (wasDisabled)
            {
                wasDisabled = false;
                Disabled = false;
            }
        }
    }
}
