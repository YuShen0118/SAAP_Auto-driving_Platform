using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerDemoNoiseScript : MonoBehaviour
    {
        public Camera RenderCamera;
        public Transform RenderQuad;

        private RenderTexture renderTexture;

        private void Start()
        {
            renderTexture = new RenderTexture(4096, 4096, 16, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Default);
            renderTexture.autoGenerateMips = false;
            renderTexture.name = "WeatherMakerDemoNoiseScript";
        }

        public void ExportClicked()
        {
            Texture2D t2d = new Texture2D(4096, 4096, TextureFormat.ARGB32, false, false);
            t2d.filterMode = FilterMode.Point;
            t2d.wrapMode = TextureWrapMode.Clamp;
            Rect rect = RenderCamera.rect;
            RenderCamera.rect = new Rect(0.0f, 0.0f, 1.0f, 1.0f);
            RenderCamera.targetTexture = renderTexture;
            RenderCamera.Render();
            RenderCamera.rect = rect;
            RenderTexture.active = renderTexture;
            t2d.ReadPixels(new Rect(0.0f, 0.0f, 4096.0f, 4096.0f), 0, 0);
            t2d.Apply();
            RenderTexture.active = null;
            RenderCamera.targetTexture = null;
            byte[] imageData = t2d.EncodeToPNG();
            Destroy(t2d);
            string docsPath = System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop);
            System.IO.File.WriteAllBytes(System.IO.Path.Combine(docsPath, "WeatherMakerNoiseTexture.png"), imageData);
        }
    }
}
