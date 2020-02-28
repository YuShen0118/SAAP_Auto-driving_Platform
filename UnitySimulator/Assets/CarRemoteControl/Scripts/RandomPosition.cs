using System;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.CrossPlatformInput;
using System.IO;
using RVO;


using Vector2 = RVO.Vector2;
using Random = System.Random;

namespace UnityStandardAssets.Vehicles.Car
{
    public class RandomPosition : MonoBehaviour
    {
        Vector3 direction;

        Random random = new Random();

        private void Awake()
        {

        }


        private void Start()
        {
            GameObject car = gameObject;
            float scale = 10;
            car.transform.position = car.transform.position +
                new Vector3((float)(random.NextDouble() * 2 - 1) * scale,
                            0,
                            (float)(random.NextDouble() * 2 - 1) * scale);

            scale = 1f;
            direction = new Vector3((float)(random.NextDouble() * 2 - 1) * scale,
                            0,
                            (float)(random.NextDouble() * 2 - 1) * scale);

        }

        private void Update()
        {
            //             GameObject car = gameObject;
            //             car.transform.position = car.transform.position + direction;
            //             if (car.transform.position.x > 90 || car.transform.position.x < -40
            //                 || car.transform.position.z > 60 || car.transform.position.z <-60)
            //             {
            //                 direction = -direction;
            //             }
        }
    }
}
