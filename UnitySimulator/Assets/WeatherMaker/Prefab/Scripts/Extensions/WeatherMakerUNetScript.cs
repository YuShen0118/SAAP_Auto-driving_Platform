using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using UnityEngine.Networking;

namespace DigitalRuby.WeatherMaker
{
    [RequireComponent(typeof(NetworkIdentity))]
    public class WeatherMakerUNetScript : NetworkBehaviour, IWeatherMakerNetworkScript
    {
        private void Cleanup()
        {
            if (WeatherMakerScript.Instance != null)
            {
                WeatherMakerScript.Instance.NetworkScript = null;
                WeatherMakerScript.Instance.WeatherProfileChanged -= WeatherProfileChanged;
            }
        }

        private void OnEnable()
        {
            if (WeatherMakerScript.Instance != null)
            {
                WeatherMakerScript.Instance.NetworkScript = this;
                WeatherMakerScript.Instance.WeatherProfileChanged += WeatherProfileChanged;
                WeatherMakerScript.Instance.WeatherManagerTransitionStarted += WeatherManagerTransitionStarted;
            }
        }

        private void OnDisable()
        {
            Cleanup();
        }

        private void OnDestroy()
        {
            Cleanup();
        }

        private void LateUpdate()
        {
            WeatherMakerScript.AssertExists();
            if (IsServer)
            {
                RpcSetTimeOfDay(WeatherMakerScript.Instance.DayNightScript.TimeOfDay);
            }
            else
            {
                // disable day night speed, the server will be syncing the day / night
                WeatherMakerScript.Instance.DayNightScript.Speed = WeatherMakerScript.Instance.DayNightScript.NightSpeed = 0.0f;
            }
        }

        [ClientRpc]
        private void RpcSetTimeOfDay(float timeOfDay)
        {
            if (!IsServer)
            {
                WeatherMakerScript.Instance.DayNightScript.TimeOfDay = timeOfDay;
            }
        }

        [ClientRpc]
        private void RpcSetProfilePath(string profilePath)
        {
            if (!IsServer)
            {
                WeatherMakerScript.Instance.WeatherProfile = Resources.Load<WeatherMakerProfileScript>(profilePath);
            }
        }

        [ClientRpc]
        private void RpcSetWeatherManagerTransition(int managerIndex, int transitionIndex, int randomSeed)
        {
            if (!IsServer)
            {
                WeatherMakerScript.Instance.ActivateWeatherManager(managerIndex);
                WeatherMakerScript.Instance.WeatherManagers[managerIndex].StartNewTransitionGroup(transitionIndex, randomSeed);
            }
        }

        private void WeatherProfileChanged(WeatherMakerProfileScript arg1, WeatherMakerProfileScript arg2)
        {
            if (IsServer)
            {
                RpcSetProfilePath(arg2.name);
            }
        }

        private void WeatherManagerTransitionStarted(int arg1, int arg2, int arg3)
        {
            if (IsServer)
            {
                RpcSetWeatherManagerTransition(arg1, arg2, arg3);
            }
        }

        public override void OnStartServer()
        {
            base.OnStartServer();
        }

        public override void OnStartClient()
        {
            base.OnStartClient();
        }

        public bool IsServer { get { return isServer; } }
    }

    public class WeatherMakerNullNetworkScript : IWeatherMakerNetworkScript
    {
        public bool IsServer { get { return true; } }
    }
}
