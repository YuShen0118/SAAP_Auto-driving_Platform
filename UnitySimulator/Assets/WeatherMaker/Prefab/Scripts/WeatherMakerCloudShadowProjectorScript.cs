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

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    [ExecuteInEditMode]
    public class WeatherMakerCloudShadowProjectorScript : MonoBehaviour
    {
        [Header("Shadow properties")]
        [Tooltip("Cloud shadow material - must not be null")]
        public Material CloudShadowMaterial;

        [Tooltip("Cloud shadow projector")]
        public Projector CloudShadowProjector;

        private Vector3 previousSunIntersect;

        private RenderTexture cloudShadowTexture;

        private Vector3 GetSunCameraIntersect(Camera c)
        {
            if (c == null)
            {
                return previousSunIntersect;
            }

            Vector3 cameraPos = c.transform.position;
            Vector3 normal = Vector3.up;
            Vector3 ray = -WeatherMakerScript.Instance.Sun.Transform.forward;
            ray.y += WeatherMakerScript.Instance.CloudScript.CloudRayOffset;
            float denom = Vector3.Dot(normal, ray);
            if (denom < Mathf.Epsilon)
            {
                // sun too low
                return previousSunIntersect;
            }

            Vector3 plane = new Vector3(cameraPos.x, WeatherMakerScript.Instance.CloudScript.CloudHeight, cameraPos.z);
            float t = Vector3.Dot(plane, normal) / denom;
            float multiplier = c.farClipPlane / t;
            multiplier = Mathf.Min(1.0f, Mathf.Pow(multiplier, 3.0f));

            if (t > c.farClipPlane * 3.0f)
            {
                // remove shadows as sun gets close to horizon
                return previousSunIntersect;
            }

            Vector3 intersect = cameraPos + (ray * t);
            CloudShadowMaterial.SetVector("_FogShadowCenterPoint", intersect);

            // reduce shadow when sun is low in the sky
            CloudShadowMaterial.SetFloat("_FogShadowMultiplier", multiplier);
            return (previousSunIntersect = intersect);
        }

        private void CreateRenderTexturesIfNeeded()
        {
            if (cloudShadowTexture == null)
            {
                cloudShadowTexture = new RenderTexture(1024, 1024, 0, RenderTextureFormat.ARGB32);
                cloudShadowTexture.name = "WeatherMakerSkySphereCloudShadowsTexture";
                cloudShadowTexture.wrapMode = TextureWrapMode.Clamp;
                cloudShadowTexture.filterMode = FilterMode.Bilinear;
                cloudShadowTexture.anisoLevel = 1;
                cloudShadowTexture.useMipMap = false;
                cloudShadowTexture.autoGenerateMips = false;
                cloudShadowTexture.mipMapBias = 0.0f;

#if DEBUG && UNITY_EDITOR

                // { GameObject debugObj = GameObject.Find("DebugQuad"); if (debugObj != null) { Renderer debugRenderer = debugObj.GetComponent<Renderer>(); if (debugRenderer != null) { debugRenderer.sharedMaterial.mainTexture = cloudShadowTexture; } } }

#endif

            }
        }

        private Vector3? ProcessCloudShadowsAndGetSunCameraIntersect(Camera c)
        {
            bool enableProjector =
            (
                WeatherMakerScript.Instance.CloudScript.CloudShadowThreshold < 0.999f &&
                WeatherMakerScript.Instance.CloudScript.CloudShadowPower < 2.0f &&
                WeatherMakerScript.Instance.CloudScript.CloudsEnabled &&
                WeatherMakerScript.Instance.Sun.LightIsOn &&
                WeatherMakerScript.Instance.Sun.Light.shadows != LightShadows.None
            );
            CreateRenderTexturesIfNeeded();
            CloudShadowProjector.enabled = enableProjector;
            CloudShadowProjector.material.SetTexture("_ShadowTex", cloudShadowTexture);
            CloudShadowProjector.orthographicSize = c.farClipPlane * WeatherMakerScript.Instance.SkySphereScript.FarClipPlaneMultiplier;
            WeatherMakerScript.Instance.CloudScript.SetShaderCloudParameters(CloudShadowMaterial);
            Vector3 pos = c.transform.position;
            pos.y = WeatherMakerScript.Instance.CloudScript.CloudHeight;
            CloudShadowProjector.transform.position = pos;
            return GetSunCameraIntersect(c);
        }

        private void OnDestroy()
        {
            if (cloudShadowTexture != null)
            {
                cloudShadowTexture.Release();
                cloudShadowTexture = null;
            }
        }

        public void RenderCloudShadows(Camera c)
        {
            Vector3? sunCameraIntersect = ProcessCloudShadowsAndGetSunCameraIntersect(c);
            if (CloudShadowProjector.enabled)
            {
                RenderTexture currentRT = RenderTexture.active;
                Graphics.SetRenderTarget(cloudShadowTexture);
                if (CloudShadowMaterial.SetPass(0))
                {
                    float scale = c.farClipPlane * WeatherMakerScript.Instance.SkySphereScript.FarClipPlaneMultiplier;
                    float viewportBounds = 1024.0f;
                    Vector3 pos = sunCameraIntersect.Value;
                    GL.PushMatrix();
                    GL.Viewport(new Rect(0.0f, 0.0f, viewportBounds, viewportBounds));
                    Vector2 bottomLeft = new Vector2(pos.x - scale, pos.z - scale);
                    Vector2 topRight = new Vector2(pos.x + scale, pos.z + scale);
                    Matrix4x4 proj = Matrix4x4.Ortho(bottomLeft.x, topRight.x, bottomLeft.y, topRight.y, -1.0f, 1.0f);
                    GL.LoadProjectionMatrix(proj);
                    GL.Begin(GL.QUADS);
                    GL.Vertex3(bottomLeft.x, bottomLeft.y, 0.0f);
                    GL.Vertex3(bottomLeft.x, topRight.y, 0.0f);
                    GL.Vertex3(topRight.x, topRight.y, 0.0f);
                    GL.Vertex3(topRight.x, bottomLeft.y, 0.0f);
                    GL.End();
                    GL.PopMatrix();
                }
                Graphics.SetRenderTarget(currentRT);
            }
        }
    }
}
