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
using UnityEngine.SceneManagement;

namespace DigitalRuby.WeatherMaker
{
    /// <summary>
    /// Represents a command buffer
    /// </summary>
    public class WeatherMakerCommandBuffer
    {
        /// <summary>
        /// Camera the command buffer is attached to
        /// </summary>
        public Camera Camera;

        /// <summary>
        /// Render queue for the command buffer
        /// </summary>
        public CameraEvent RenderQueue;

        /// <summary>
        /// The command buffer
        /// </summary>
        public CommandBuffer CommandBuffer;

        /// <summary>
        /// A copy of the original material to render with, will be destroyed when command buffer is removed
        /// </summary>
        public Material Material;

        /// <summary>
        /// Optional action to update material properties
        /// </summary>
        public System.Action<WeatherMakerCommandBuffer> UpdateMaterial;
    }

    public class WeatherMakerCommandBufferManagerScript : MonoBehaviour
    {
        private readonly List<WeatherMakerCommandBuffer> commandBuffers = new List<WeatherMakerCommandBuffer>();
        private readonly Vector4[] frustumCorners = new Vector4[8];
        private readonly Vector3[] frustumCorners2 = new Vector3[4];

        private void UpdateShaderProperties()
        {
            foreach (WeatherMakerCommandBuffer c in commandBuffers)
            {
                Transform ct = c.Camera.transform;
                c.Camera.CalculateFrustumCorners(c.Camera.rect, c.Camera.farClipPlane, c.Camera.stereoEnabled ? Camera.MonoOrStereoscopicEye.Left : Camera.MonoOrStereoscopicEye.Mono, frustumCorners2);
                // bottom left, top left, bottom right, top right
                frustumCorners[0] = ct.TransformDirection(frustumCorners2[0]);
                frustumCorners[1] = ct.TransformDirection(frustumCorners2[1]);
                frustumCorners[2] = ct.TransformDirection(frustumCorners2[3]);
                frustumCorners[3] = ct.TransformDirection(frustumCorners2[2]);
                c.Camera.CalculateFrustumCorners(c.Camera.rect, c.Camera.farClipPlane, Camera.MonoOrStereoscopicEye.Right, frustumCorners2);
                // bottom left, top left, bottom right, top right
                frustumCorners[4] = ct.TransformDirection(frustumCorners2[0]);
                frustumCorners[5] = ct.TransformDirection(frustumCorners2[1]);
                frustumCorners[6] = ct.TransformDirection(frustumCorners2[3]);
                frustumCorners[7] = ct.TransformDirection(frustumCorners2[2]);
                c.Material.SetVectorArray("_WeatherMakerCameraFrustumRays", frustumCorners);
                if (WeatherMakerScript.Instance != null && WeatherMakerScript.Instance.multiPassStereoRenderingEnabled)
                {
                    if (!c.Material.IsKeywordEnabled("UNITY_MULTI_PASS_STEREO"))
                    {
                        c.Material.EnableKeyword("UNITY_MULTI_PASS_STEREO");
                    }
                }
                else if (c.Material.IsKeywordEnabled("UNITY_MULTI_PASS_STEREO"))
                {
                    c.Material.DisableKeyword("UNITY_MULTI_PASS_STEREO");
                }
                if (c.UpdateMaterial != null)
                {
                    c.UpdateMaterial(c);
                }
            }
        }

        private void CleanupCommandBuffer(WeatherMakerCommandBuffer commandBuffer)
        {
            if (commandBuffer == null)
            {
                return;
            }
            else if (commandBuffer.Material != null)
            {
                GameObject.Destroy(commandBuffer.Material);
            }
            if (commandBuffer.Camera != null && commandBuffer.CommandBuffer != null)
            {
                commandBuffer.Camera.RemoveCommandBuffer(commandBuffer.RenderQueue, commandBuffer.CommandBuffer);
                commandBuffer.CommandBuffer.Release();
            }
        }

        private void CleanupCameras()
        {
            // remove destroyed camera command buffers
            for (int i = commandBuffers.Count - 1; i >= 0; i--)
            {
                if (commandBuffers[i].Camera == null)
                {
                    CleanupCommandBuffer(commandBuffers[i]);
                    commandBuffers.RemoveAt(i);
                }
            }
        }

        private void RemoveAllCommandBuffers()
        {
            for (int i = commandBuffers.Count - 1; i >= 0; i--)
            {
                CleanupCommandBuffer(commandBuffers[i]);
            }
            commandBuffers.Clear();
        }

        private void SceneManagerSceneLoaded(Scene newScene, LoadSceneMode mode)
        {
            RemoveAllCommandBuffers();
        }

        private void Start()
        {
            UnityEngine.SceneManagement.SceneManager.sceneLoaded += SceneManagerSceneLoaded;
        }

        private void LateUpdate()
        {
            CleanupCameras();
            UpdateShaderProperties();
        }

        /// <summary>
        /// Add a command buffer
        /// </summary>
        /// <param name="commandBuffer">Command buffer to add, the CommandBuffer property must have a unique name assigned</param>
        /// <returns>True if added, false if not</returns>
        public bool AddCommandBuffer(WeatherMakerCommandBuffer commandBuffer)
        {
            if (commandBuffer == null || string.IsNullOrEmpty(commandBuffer.CommandBuffer.name))
            {
                return false;
            }
            RemoveCommandBuffer(commandBuffer.Camera, commandBuffer.CommandBuffer.name);
            commandBuffers.Add(commandBuffer);
            commandBuffer.Camera.AddCommandBuffer(commandBuffer.RenderQueue, commandBuffer.CommandBuffer);
            return true;
        }

        /// <summary>
        /// Remove a command buffer
        /// </summary>
        /// <param name="commandBuffer">Command buffer to remove</param>
        /// <returns>True if removed, false if not</returns>
        public bool RemoveCommandBuffer(WeatherMakerCommandBuffer commandBuffer)
        {
            if (commandBuffer == null)
            {
                return false;
            }
            int index = commandBuffers.IndexOf(commandBuffer);
            if (index >= 0)
            {
                CleanupCommandBuffer(commandBuffers[index]);
                commandBuffers.RemoveAt(index);
                return true;
            }
            return false;
        }

        /// <summary>
        /// Remove a command buffer
        /// </summary>
        /// <param name="camera">Camera to remove command buffer on</param>
        /// <param name="name">Name of the command buffer to remove</param>
        /// <returns>True if removed, false if not</returns>
        public bool RemoveCommandBuffer(Camera camera, string name)
        {
            if (camera == null || string.IsNullOrEmpty(name))
            {
                return false;
            }
            for (int i = 0; i < commandBuffers.Count; i++)
            {
                if (commandBuffers[i].Camera == camera && commandBuffers[i].CommandBuffer.name == name)
                {
                    CleanupCommandBuffer(commandBuffers[i]);
                    commandBuffers.RemoveAt(i);
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Remove all command buffers with a specified name
        /// </summary>
        /// <param name="name">Name of the command buffers to remove</param>
        /// <returns>True if at least one command buffer removed, false otherwise</returns>
        public bool RemoveCommandBuffers(string name)
        {
            bool foundOne = false;
            for (int i = commandBuffers.Count - 1; i >= 0; i--)
            {
                if (commandBuffers[i].CommandBuffer.name == name)
                {
                    CleanupCommandBuffer(commandBuffers[i]);
                    commandBuffers.RemoveAt(i);
                    foundOne = true;
                }
            }
            return foundOne;
        }

        /// <summary>
        /// Checks for existance of a command buffer
        /// </summary>
        /// <param name="commandBuffer">Command buffer to check for</param>
        /// <returns>True if exists, false if not</returns>
        public bool ContainsCommandBuffer(WeatherMakerCommandBuffer commandBuffer)
        {
            if (commandBuffer == null || commandBuffer.Camera == null)
            {
                return false;
            }
            foreach (CommandBuffer cameraCommandBuffer in commandBuffer.Camera.GetCommandBuffers(commandBuffer.RenderQueue))
            {
                if (commandBuffer.CommandBuffer == cameraCommandBuffer)
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Checks for existance of a command buffer by camera and name
        /// </summary>
        /// <param name="camera">Camera to check for</param>
        /// <param name="renderQueue">Camera event to check for</param>
        /// <param name="name">Name to check for</param>
        /// <returns>True if exists, false if not</returns>
        public bool ContainsCommandBuffer(Camera camera, CameraEvent renderQueue, string name)
        {
            if (camera == null || string.IsNullOrEmpty(name))
            {
                return false;
            }
            foreach (CommandBuffer cameraCommandBuffer in camera.GetCommandBuffers(renderQueue))
            {
                if (cameraCommandBuffer.name == name)
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Added to allow setting camera / shader properties at the last possible moment before rendering
        /// </summary>
        /// <param name="c">Camera to pre-cull</param>
        public void PreCullCamera(Camera c)
        {
            UpdateShaderProperties();
        }
    }
}