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
    [System.Serializable]
    public class WeatherMakerFullScreenEffect : System.IDisposable
    {
        [Tooltip("Render queue for this full screen effect. Do not change this at runtime, set it in the inspector once.")]
        public CameraEvent RenderQueue = CameraEvent.BeforeForwardAlpha;

        [Tooltip("Material for rendering/creating the effect")]
        public Material Material;

        [Tooltip("Material for blurring")]
        public Material BlurMaterial;

        [Tooltip("Material to render the final pass if needed, not all setups will need this but it should be set anyway")]
        public Material BlitMaterial;

        [Tooltip("Downsample scale for Material")]
        [Range(0.01f, 1.0f)]
        public float DownsampleScale = 1.0f;
        private float lastDownSampleScale = -1.0f;

        [Tooltip("Downsample scale for render buffer sampling, or 0 to not sample the render buffer.")]
        [Range(0.0f, 1.0f)]
        public float DownsampleRenderBufferScale = 0.0f;
        private float lastDownsampleRenderBufferScale = -1.0f;

        [Tooltip("Name of the texture to set if sampling the render buffer")]
        public string DownsampleRenderBufferTextureName;

        [Tooltip("Blur shader type")]
        public BlurShaderType BlurShaderType = BlurShaderType.None;
        private BlurShaderType lastBlurShaderType = (BlurShaderType)0x7FFFFFFF;

        [Tooltip("ZTest")]
        public UnityEngine.Rendering.CompareFunction ZTest = CompareFunction.Always;

        /// <summary>
        /// The name for the command buffer that will be created for this effect. This should be unique for your project.
        /// </summary>
        public string CommandBufferName { get; set; }

        /// <summary>
        /// Whether the effect is enabled. The effect can be disabled to prevent command buffers from being created.
        /// </summary>
        public bool Enabled { get; set; }

        /// <summary>
        /// Action to fire when Material properties should be updated
        /// </summary>
        public System.Action<WeatherMakerCommandBuffer> UpdateMaterialProperties { get; set; }

        private bool needsToBeRecreated;
        private readonly List<Camera> cameras = new List<Camera>();

        private void CreateCommandBuffer(Camera camera)
        {
            CommandBuffer commandBuffer = new CommandBuffer { name = CommandBufferName };
            //commandBuffer.SetViewMatrix(Matrix4x4.Ortho(0.0f, 1.0f, 0.0f, 1.0f, camera.farClipPlane, camera.farClipPlane));
            int frameBufferSourceId = -1;
            RenderTargetIdentifier frameBufferSource = new RenderTargetIdentifier(BuiltinRenderTextureType.None);
            Material material = new Material(Material);

            if (DownsampleRenderBufferScale > 0.0f)
            {
                // render camera target to texture, performing separate down-sampling
                frameBufferSourceId = Shader.PropertyToID(DownsampleRenderBufferTextureName);
                frameBufferSource = new RenderTargetIdentifier(frameBufferSourceId);
                commandBuffer.GetTemporaryRT(frameBufferSourceId, (int)(Screen.width * DownsampleRenderBufferScale), (int)(Screen.height * DownsampleRenderBufferScale), 0, FilterMode.Bilinear, RenderTextureFormat.ARGB32);
                commandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, frameBufferSource, null);
            }

            if (BlurShaderType == BlurShaderType.None && DownsampleScale >= 0.99f)
            {
                // draw directly to render target
                material.SetInt("_SrcBlendMode", (int)BlendMode.One);
                material.SetInt("_DstBlendMode", (int)BlendMode.OneMinusSrcAlpha);
                material.SetInt("_ZTest", (int)ZTest);
                commandBuffer.Blit(frameBufferSource, new RenderTargetIdentifier(BuiltinRenderTextureType.CameraTarget), material);
            }
            else
            {
                // render to texture, using current image target as input _MainTex, no blend
                int intermediateRenderTargetId = Shader.PropertyToID("_MainTex");
                RenderTargetIdentifier intermediateRenderTarget = new RenderTargetIdentifier(intermediateRenderTargetId);
                commandBuffer.GetTemporaryRT(intermediateRenderTargetId, (int)(Screen.width * DownsampleScale), (int)(Screen.height * DownsampleScale), 0, FilterMode.Bilinear, RenderTextureFormat.ARGB32);
                material.SetInt("_SrcBlendMode", (int)BlendMode.One);
                material.SetInt("_DstBlendMode", (int)BlendMode.Zero);
                material.SetInt("_ZTest", (int)ZTest);
                commandBuffer.Blit(frameBufferSource, intermediateRenderTarget, material);

                // blur if requested
                if (BlurShaderType == BlurShaderType.None)
                {
                    // blit texture directly on top of camera, alpha blend
                    BlitMaterial.SetInt("_SrcBlendMode", (int)BlendMode.One);
                    BlitMaterial.SetInt("_DstBlendMode", (int)BlendMode.OneMinusSrcAlpha);
                    BlitMaterial.SetInt("_ZTest", (int)ZTest);
                    commandBuffer.Blit(intermediateRenderTarget, BuiltinRenderTextureType.CameraTarget, BlitMaterial);
                }
                else
                {
                    // blur texture directly on to camera target, alpha blend
                    BlurMaterial.SetInt("_SrcBlendMode", (int)BlendMode.One);
                    BlurMaterial.SetInt("_DstBlendMode", (int)BlendMode.OneMinusSrcAlpha);
                    if (BlurShaderType == BlurShaderType.GaussianBlur7)
                    {
                        BlurMaterial.EnableKeyword("BLUR7");
                    }
                    else
                    {
                        BlurMaterial.DisableKeyword("BLUR7");
                    }
                    BlurMaterial.SetInt("_ZTest", (int)ZTest);
                    commandBuffer.Blit(intermediateRenderTarget, BuiltinRenderTextureType.CameraTarget, BlurMaterial);
                }

                // cleanup
                commandBuffer.ReleaseTemporaryRT(intermediateRenderTargetId);
            }

            if (DownsampleRenderBufferScale > 0.0f)
            {
                // cleanup
                commandBuffer.ReleaseTemporaryRT(frameBufferSourceId);
            }

            // add to manager
            WeatherMakerCommandBuffer c = new WeatherMakerCommandBuffer
            {
                Camera = camera,
                CommandBuffer = commandBuffer,
                Material = material,
                RenderQueue = RenderQueue,
                UpdateMaterial = UpdateMaterialProperties
            };

            WeatherMakerScript.Instance.CommandBufferManagerScript.AddCommandBuffer(c);
            cameras.Add(camera);
        }

        /// <summary>
        /// Call from LateUpdate in script
        /// </summary>
        public void LateUpdate()
        {
            if (Enabled)
            {
                needsToBeRecreated = lastDownSampleScale != DownsampleScale ||
                    lastDownsampleRenderBufferScale != DownsampleRenderBufferScale ||
                    lastBlurShaderType != BlurShaderType;
                lastDownSampleScale = DownsampleScale;
                lastDownsampleRenderBufferScale = DownsampleRenderBufferScale;
                lastBlurShaderType = BlurShaderType;
            }
            else
            {
                Dispose();
            }
        }

        /// <summary>
        /// Update the full screen effect, usually called from OnPreRender or OnPreCull
        /// </summary>
        /// <param name="camera">Camera</param>
        public void SetupCamera(Camera camera)
        {
            if (Enabled && (needsToBeRecreated || !WeatherMakerScript.Instance.CommandBufferManagerScript.ContainsCommandBuffer(camera, RenderQueue, CommandBufferName)))
            {
                CreateCommandBuffer(camera);
            }
        }

        /// <summary>
        /// Cleanup all resources and set Enabled to false
        /// </summary>
        public void Dispose()
        {
            lastDownSampleScale = -1.0f;
            lastDownsampleRenderBufferScale = -1.0f;
            lastBlurShaderType = (BlurShaderType)0x7FFFFFFF;
            foreach (Camera c in cameras)
            {
                if (c != null)
                {
                    WeatherMakerScript.Instance.CommandBufferManagerScript.RemoveCommandBuffer(c, CommandBufferName);
                }
            }
            cameras.Clear();
            Enabled = false;
        }
    }
}