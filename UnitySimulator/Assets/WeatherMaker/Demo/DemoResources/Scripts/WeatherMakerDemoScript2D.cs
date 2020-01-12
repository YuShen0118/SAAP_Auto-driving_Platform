﻿//
// Weather Maker for Unity
// (c) 2016 Digital Ruby, LLC
// Source code may be used for personal or commercial projects.
// Source code may NOT be redistributed or sold.
// 

using UnityEngine;
using System.Collections;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerDemoScript2D : MonoBehaviour
    {
        private void Start()
        {

        }

        private void Update()
        {
            Vector3 worldBottomLeft = Camera.main.ViewportToWorldPoint(Vector3.zero);
            Vector3 worldTopRight = Camera.main.ViewportToWorldPoint(Vector3.one);
            float visibleWorldWidth = worldTopRight.x - worldBottomLeft.x;

            if (Input.GetKey(KeyCode.LeftArrow))
            {
                Camera.main.transform.Translate(Time.deltaTime * -(visibleWorldWidth * 0.1f), 0.0f, 0.0f);
            }
            else if (Input.GetKey(KeyCode.RightArrow))
            {
                Camera.main.transform.Translate(Time.deltaTime * (visibleWorldWidth * 0.1f), 0.0f, 0.0f);
            }
        }
    }
}