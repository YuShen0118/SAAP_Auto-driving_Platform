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
using System.Collections;
using System;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerLegacyCloudScript2D : MonoBehaviour
    {
        [Header("Rendering")]
        [Tooltip("Material for the clouds")]
        public Material Material;
        private Material sharedMaterial;
        private Material lastMaterial;

        [Tooltip("Override the texture for the material")]
        public Texture2D MaterialTexture;

        [Tooltip("Override the mask texture for the material")]
        public Texture2D MaterialMaskTexture;

        [Tooltip("Tint color for the clouds")]
        public Color TintColor = Color.white;

        [Range(1, 16)]
        [Tooltip("Number of rows in the cloud material")]
        public int MaterialRows = 1;

        [Range(1, 16)]
        [Tooltip("Number of columns in the cloud material")]
        public int MaterialColumns = 1;

        [Header("Positioning")]
        [Tooltip("Whether to anchor the clouds to the Anchor (i.e. main camera) or not")]
        public bool AnchorClouds = true;

        [Tooltip("Offset from the anchor that the clouds should position themselves at")]
        public Vector3 AnchorOffset = new Vector3(0.0f, -700.0f, 0.0f);

        [Header("Count")]
        [Range(0, 16250)]
        [Tooltip("The total number of clouds to create")]
        public int NumberOfClouds = 1000;

        private ParticleSystem cloudParticleSystem;
        private Renderer cloudParticleSystemRenderer;

        private void UpdateParticleSystem()
        {
            var m = cloudParticleSystem.main;
            m.maxParticles = NumberOfClouds;
            var anim = cloudParticleSystem.textureSheetAnimation;
            anim.numTilesX = MaterialColumns;
            anim.numTilesY = MaterialRows;
            cloudParticleSystemRenderer.sharedMaterial.mainTexture = MaterialTexture;
            cloudParticleSystemRenderer.sharedMaterial.SetColor("_TintColor", TintColor);
            cloudParticleSystemRenderer.sharedMaterial.EnableKeyword("ORTHOGRAPHIC_MODE");
        }

        private void UpdateTransform()
        {
            if (WeatherMakerScript.Instance.Camera != null && AnchorClouds)
            {
                Vector3 pos = WeatherMakerScript.Instance.Camera.transform.position;
                Vector3 curPos = gameObject.transform.position;
                curPos.x = pos.x + AnchorOffset.x;
                curPos.y = AnchorOffset.y;
                curPos.z = (WeatherMakerScript.Instance.CameraIsOrthographic ? curPos.z : pos.z + AnchorOffset.z);
                gameObject.transform.position = curPos;
            }
        }

        private void UpdateMaterial()
        {
            if (Material != lastMaterial)
            {
                Renderer renderer = GetComponent<Renderer>();
                sharedMaterial = (Material == null ? null : new Material(Material));
                if (renderer != null)
                {
                    renderer.sharedMaterial = sharedMaterial;
                }
                lastMaterial = Material;
            }
            if (sharedMaterial != null)
            {
                sharedMaterial.mainTexture = (MaterialTexture == null ? sharedMaterial.mainTexture : MaterialTexture);
                sharedMaterial.SetTexture("_MaskTex", MaterialMaskTexture);
                sharedMaterial.SetColor("_TintColor", TintColor);
            }
        }

        private void Start()
        {
            UpdateMaterial();
            UpdateTransform();
            cloudParticleSystem = GetComponentInChildren<ParticleSystem>();
            cloudParticleSystemRenderer = cloudParticleSystem.GetComponent<Renderer>();
            UpdateParticleSystem();
        }

        private void Update()
        {
            UpdateMaterial();
            UpdateTransform();
            UpdateParticleSystem();
        }

        public void CreateClouds()
        {
            cloudParticleSystem.Play();
        }

        public void RemoveClouds()
        {
            cloudParticleSystem.Stop();
        }

        public void Reset()
        {
            cloudParticleSystem.Stop();
            cloudParticleSystem.Clear();
        }
    }
}