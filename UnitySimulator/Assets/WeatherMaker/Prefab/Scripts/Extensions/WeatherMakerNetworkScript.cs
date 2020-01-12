using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    public interface IWeatherMakerNetworkScript
    {
        bool IsServer { get; }
    }
}
