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

using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerConfigurationScript : MonoBehaviour
    {
        public bool ShowFPS = true;
        public bool ShowTimeOfDay = true;
        public GameObject ConfigurationPanel;
        public UnityEngine.UI.Text LabelFPS;
        public UnityEngine.UI.Slider TransitionDurationSlider;
        public UnityEngine.UI.Slider IntensitySlider;
        public UnityEngine.UI.Toggle MouseLookEnabledCheckBox;
        public UnityEngine.UI.Toggle FlashlightToggle;
        public UnityEngine.UI.Toggle TimeOfDayEnabledCheckBox;
        public UnityEngine.UI.Toggle CloudToggle2D;
        public UnityEngine.UI.Toggle CollisionToggle;
        public UnityEngine.UI.Slider DawnDuskSlider;
        public UnityEngine.UI.Text TimeOfDayText;
        public UnityEngine.UI.Dropdown CloudDropdown;
        public UnityEngine.EventSystems.EventSystem EventSystem;
        public GameObject SidePanel;

        private int frameCount = 0;
        private float nextFrameUpdate = 0.0f;
        private float fps = 0.0f;
        private float frameUpdateRate = 4.0f; // 4 per second
        private int frameCounter;

        private void UpdateTimeOfDay()
        {
            DawnDuskSlider.value = WeatherMakerScript.Instance.TimeOfDay;
            if (TimeOfDayText.IsActive() && ShowTimeOfDay)
            {
                TimeSpan t = TimeSpan.FromSeconds(WeatherMakerScript.Instance.TimeOfDay);
                TimeOfDayText.text = string.Format("{0:00}:{1:00}:{2:00}", t.Hours, t.Minutes, t.Seconds);
            }
        }

        private void DisplayFPS()
        {
            if (LabelFPS != null && ShowFPS)
            {
                frameCount++;
                if (Time.time > nextFrameUpdate)
                {
                    nextFrameUpdate += (1.0f / frameUpdateRate);
                    fps = (int)Mathf.Floor((float)frameCount * frameUpdateRate);
                    LabelFPS.text = "FPS: " + fps;
                    frameCount = 0;
                }
            }
        }

        private void Start()
        {
            IntensitySlider.value = WeatherMakerScript.Instance.PrecipitationIntensity;
            DawnDuskSlider.value = WeatherMakerScript.Instance.TimeOfDay;
            nextFrameUpdate = Time.time;
            CollisionToggle.isOn = WeatherMakerScript.Instance.PrecipitationCollisionEnabled;
            if (UnityEngine.EventSystems.EventSystem.current == null && ConfigurationPanel != null && ConfigurationPanel.activeInHierarchy)
            {
                EventSystem.gameObject.SetActive(true);
                UnityEngine.EventSystems.EventSystem.current = EventSystem;
            }
        }

        private void Update()
        {
            if (WeatherMakerScript.Instance.Camera == null)
            {
                return;
            }
            
            DisplayFPS();
            if (Input.GetKeyDown(KeyCode.B))
            {
                LightningStrikeButtonClicked();
            }
            if (Input.GetKeyDown(KeyCode.BackQuote) && (Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl)))
            {
                // Unity bug, disabling or setting transform scale to 0 will break any projectors in the scene
                RectTransform r = SidePanel.GetComponent<RectTransform>();
                Vector2 ap = r.anchoredPosition;
                if (r.anchoredPosition.x < 0.0f)
                {
                    ap.x = 110.0f;
                }
                else
                {
                    ap.x = -9999.0f;
                }
                r.anchoredPosition = ap;
            }
            if (Input.GetKeyDown(KeyCode.Escape) && Input.GetKey(KeyCode.LeftShift))
            {
                UnityEngine.SceneManagement.SceneManager.LoadScene(0);
            }
            UpdateTimeOfDay();
            frameCounter++;
        }

        // Weather configuration...

        public void RainToggleChanged(bool isOn)
        {
            WeatherMakerScript.Instance.Precipitation = (isOn ? WeatherMakerPrecipitationType.Rain : WeatherMakerPrecipitationType.None);
        }

        public void SnowToggleChanged(bool isOn)
        {
            WeatherMakerScript.Instance.Precipitation = (isOn ? WeatherMakerPrecipitationType.Snow : WeatherMakerPrecipitationType.None);
        }

        public void HailToggleChanged(bool isOn)
        {
            WeatherMakerScript.Instance.Precipitation = (isOn ? WeatherMakerPrecipitationType.Hail : WeatherMakerPrecipitationType.None);
        }

        public void SleetToggleChanged(bool isOn)
        {
            WeatherMakerScript.Instance.Precipitation = (isOn ? WeatherMakerPrecipitationType.Sleet : WeatherMakerPrecipitationType.None);
        }

        public void CloudToggleChanged()
        {
            WeatherMakerScript.Instance.CloudChangeDuration = TransitionDurationSlider.value;
            if (CloudDropdown == null)
            {
                // 2D only supports storm clouds
                WeatherMakerScript.Instance.Clouds = (CloudToggle2D.isOn ? WeatherMakerCloudType.Storm : WeatherMakerCloudType.None);
            }
            else if (CloudDropdown.value == 1)
            {
                WeatherMakerScript.Instance.Clouds = WeatherMakerCloudType.Light;
            }
            else if (CloudDropdown.value == 2)
            {
                WeatherMakerScript.Instance.Clouds = WeatherMakerCloudType.Medium;
            }
            else if (CloudDropdown.value == 3)
            {
                WeatherMakerScript.Instance.Clouds = WeatherMakerCloudType.Heavy;
            }
            else if (CloudDropdown.value == 4)
            {
                WeatherMakerScript.Instance.Clouds = WeatherMakerCloudType.HeavyBright;
            }
            else if (CloudDropdown.value == 5)
            {
                WeatherMakerScript.Instance.Clouds = WeatherMakerCloudType.Storm;
            }
            else
            {
                WeatherMakerScript.Instance.Clouds = WeatherMakerCloudType.None;
            }
        }

        public void LightningToggleChanged(bool isOn)
        {
            WeatherMakerScript.Instance.LightningScript.EnableLightning = isOn;
        }

        public void CollisionToggleChanged(bool isOn)
        {
            WeatherMakerScript.Instance.PrecipitationCollisionEnabled = isOn;
        }

        public void WindToggleChanged(bool isOn)
        {
            WeatherMakerScript.Instance.WindScript.AnimateWindIntensity(isOn ? 0.3f : 0.0f, TransitionDurationSlider.value);
        }

        public void TransitionDurationSliderChanged(float val)
        {
            WeatherMakerScript.Instance.PrecipitationChangeDuration = val;
        }

        public void IntensitySliderChanged(float val)
        {
            WeatherMakerScript.Instance.PrecipitationIntensity = val;
        }

        public void MouseLookEnabledChanged(bool val)
        {
            MouseLookEnabledCheckBox.isOn = val;
            foreach (GameObject obj in GameObject.FindGameObjectsWithTag("Player"))
            {
                UnityEngine.Networking.NetworkBehaviour network = obj.GetComponent<UnityEngine.Networking.NetworkBehaviour>();
                if (network == null || network.isLocalPlayer)
                {
                    WeatherMakerPlayerControllerScript controller = obj.GetComponent<WeatherMakerPlayerControllerScript>();
                    if (controller != null)
                    {
                        controller.EnableMouseLook = val;
                    }
                }
            }
        }

        public void FlashlightChanged(bool val)
        {
            foreach (GameObject obj in GameObject.FindGameObjectsWithTag("Player"))
            {
                UnityEngine.Networking.NetworkBehaviour network = obj.GetComponent<UnityEngine.Networking.NetworkBehaviour>();
                if (network == null || network.isLocalPlayer)
                {
                    Light[] lights = obj.GetComponentsInChildren<Light>();
                    foreach (Light light in lights)
                    {
                        if (light.name == "Flashlight")
                        {
                            light.enabled = val;
                            break;
                        }
                    }
                }
            }
            if (FlashlightToggle != null)
            {
                FlashlightToggle.isOn = val;
            }
        }

        public void FogChanged(bool val)
        {
            // if fog is not active, set the start fog density to 0, otherwise start at whatever density it is at
            float startFogDensity = WeatherMakerScript.Instance.FogScript.FogDensity;
            float endFogDensity = (startFogDensity == 0.0f ? 0.02f : 0.0f);
            WeatherMakerScript.Instance.FogScript.TransitionFogDensity(startFogDensity, endFogDensity, TransitionDurationSlider.value);
        }

        public void ManagerChanged(bool val)
        {
            if (WeatherMakerScript.Instance.WeatherManagers != null && WeatherMakerScript.Instance.WeatherManagers.Count > 0)
            {
                WeatherMakerScript.Instance.WeatherManagers[0].gameObject.SetActive(val);
            }
        }

        public void TimeOfDayEnabledChanged(bool val)
        {
            WeatherMakerScript.Instance.DayNightScript.Speed = WeatherMakerScript.Instance.DayNightScript.NightSpeed = (val ? 10.0f : 0.0f);
        }

        public void LightningStrikeButtonClicked()
        {
            WeatherMakerScript.Instance.LightningScript.CallIntenseLightning();
        }

        public void DawnDuskSliderChanged(float val)
        {
            WeatherMakerScript.Instance.DayNightScript.TimeOfDay = val;
        }
    }
}