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

using UnityEngine;
using System.Collections.Generic;

namespace DigitalRuby.WeatherMaker
{
    [RequireComponent(typeof(ParticleSystem))]
    public class WeatherMakerParticleCollisionScript : MonoBehaviour
    {
        private readonly List<ParticleCollisionEvent> collisionEvents = new List<ParticleCollisionEvent>();
        private ParticleSystem collisionParticleSystem;

        [Tooltip("Explosion particle system")]
        public ParticleSystem ExplosionParticleSystem;

        private void Start()
        {
            collisionParticleSystem = GetComponent<ParticleSystem>(); 
        }

        private void Update()
        {

        }
        
        private void OnParticleCollision(GameObject obj)
        {
            if (ExplosionParticleSystem != null)
            {
                int count = collisionParticleSystem.GetCollisionEvents(obj, collisionEvents);
                var em = ExplosionParticleSystem.emission;
                var rate = em.rateOverTime;
                for (int i = 0; i < count; i++)
                {
                    ParticleCollisionEvent evt = collisionEvents[i];
                    Vector3 pos = evt.intersection;
                    ExplosionParticleSystem.transform.position = pos;
                    ExplosionParticleSystem.Emit(UnityEngine.Random.Range((int)rate.constantMin, (int)rate.constantMax + 1));
                }
            }
        }
    }
}