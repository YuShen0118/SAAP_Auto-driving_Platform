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
using UnityEngine.Rendering;

namespace DigitalRuby.WeatherMaker
{
    [RequireComponent(typeof(Light))]
    public class WeatherMakerShadowMapScript : MonoBehaviour
    {
        [Tooltip("The texture name for shaders to access the shadow map")]
        public string ShaderTextureName;

        /*
        [Tooltip("Material to down-size the shadow map")]
        public Material ShadowCopyMaterial;

        [Tooltip("Resolution of the shadow map. Smaller resolutions are more performant but less quality.")]
        [Range(256, 4096)]
        public int ShadowmapResolution = 1024;
        private int lastShadowmapResolution = -1;

        [Tooltip("Type of blur to use as shadow map is down-sampled.")]
        public BlurShaderType ShadowmapBlur = BlurShaderType.GaussianBlur17;
        private BlurShaderType lastShadowmapBlur = BlurShaderType.None;

        public RenderTexture ShadowMapCopy { get; private set; }
        */

        private Light _light;
        private CommandBuffer _commandBuffer;

        private void CreateCommandBuffer()
        {
            if (_light == null)
            {
                return;
            }
            else if (_commandBuffer != null)
            {
                try
                {
                    _light.RemoveCommandBuffer(LightEvent.AfterShadowMap, _commandBuffer);
                }
                catch
                {
                    // eat exceptions
                }
            }

            RenderTargetIdentifier renderedShadowmapIdentifier = BuiltinRenderTextureType.CurrentActive;
            _commandBuffer = new CommandBuffer { name = "WeatherMakerShadowMapScript" };
            _commandBuffer.SetGlobalTexture(ShaderTextureName, renderedShadowmapIdentifier);
            _light.AddCommandBuffer(LightEvent.AfterShadowMap, _commandBuffer);

            /*
            _commandBuffer.SetShadowSamplingMode(renderedShadowmapIdentifier, ShadowSamplingMode.RawDepth);
            ShadowMapCopy = new RenderTexture(ShadowmapResolution, ShadowmapResolution, 0, RenderTextureFormat.RFloat);
            ShadowMapCopy.filterMode = FilterMode.Bilinear;
            ShadowMapCopy.wrapMode = TextureWrapMode.Clamp;
            ShadowMapCopy.useMipMap = false;
            ShadowMapCopy.autoGenerateMips = false;
            ShadowMapCopy.mipMapBias = 0.0f;
            ShadowMapCopy.Create();
            RenderTargetIdentifier shadowMapCopyIdentifier = new RenderTargetIdentifier(ShadowMapCopy);
            _commandBuffer.SetGlobalTexture("_MainTex", renderedShadowmapIdentifier);
            Vector4 texelSize = new Vector4(1.0f / ShadowmapResolution, 1.0f / ShadowmapResolution, ShadowmapResolution, ShadowmapResolution);
            _commandBuffer.SetGlobalVector("_WeatherMakerShadowMap_TexelSize", texelSize);
            Shader.SetGlobalVector("_WeatherMakerShadowMap_TexelSize", texelSize);
            Shader.SetGlobalTexture(ShaderTextureName, ShadowMapCopy);
            ShadowCopyMaterial.DisableKeyword("BLUR7");
            ShadowCopyMaterial.DisableKeyword("BLUR17");

            if (ShadowmapBlur == BlurShaderType.GaussianBlur17)
            {
                ShadowCopyMaterial.EnableKeyword("BLUR17");
                _commandBuffer.Blit(renderedShadowmapIdentifier, shadowMapCopyIdentifier, ShadowCopyMaterial, -1);
            }
            else if (ShadowmapBlur == BlurShaderType.GaussianBlur7)
            {
                ShadowCopyMaterial.EnableKeyword("BLUR7");
                _commandBuffer.Blit(renderedShadowmapIdentifier, shadowMapCopyIdentifier, ShadowCopyMaterial, -1);
            }
            else
            {
                _commandBuffer.Blit(renderedShadowmapIdentifier, shadowMapCopyIdentifier, ShadowCopyMaterial, 0);
            }
            
#if DEBUG && UNITY_EDITOR

            { GameObject debugObj = GameObject.Find("DebugQuad"); if (debugObj != null) { Renderer debugRenderer = debugObj.GetComponent<Renderer>(); if (debugRenderer != null) { debugRenderer.sharedMaterial.mainTexture = ShadowMapCopy; } } }

#endif
            */

            //lastShadowmapResolution = ShadowmapResolution;
            //lastShadowmapBlur = ShadowmapBlur;
        }

        private void Start()
        {
            _light = GetComponent<Light>();
            CreateCommandBuffer();
        }

        private void Update()
        {
            //if (lastShadowmapResolution != ShadowmapResolution || lastShadowmapBlur != ShadowmapBlur)
            //{
                //CreateCommandBuffer();
            //}
        }

        private void OnEnable()
        {
            CreateCommandBuffer();
        }

        private void OnDisable()
        {
            if (_light != null && _commandBuffer != null)
            {
                _light.RemoveCommandBuffer(LightEvent.AfterShadowMap, _commandBuffer);
                _commandBuffer.Release();
            }
        }

        private void OnDestroy()
        {
            if (_light != null && _commandBuffer != null)
            {
                try
                {
                    _light.RemoveCommandBuffer(LightEvent.AfterShadowMap, _commandBuffer);
                }
                catch
                {
                    // eat exception
                }
                _commandBuffer.Release();
            }
            //if (ShadowMapCopy != null)
            //{
                //ShadowMapCopy.Release();
            //}
        }
    }
}