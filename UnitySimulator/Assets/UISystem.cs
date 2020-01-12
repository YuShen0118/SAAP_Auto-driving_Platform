using System.Collections;
using System.Collections.Generic;
using UnityStandardAssets.Vehicles.Car;
using UnityEngine;
using UnityEngine.UI;

public class UISystem : MonoBehaviour {

    public CarController carController;
    public Text SafetyText;
    public Text TTCText;
    public Text SteeringAngleText;
    public Text SpeedText;
    public Text DataCaptureText;
    public RawImage CameraImage;
    //public Image SpeedAnimation;
    private Camera CenterCamera;

    void Start () {

        CenterCamera = GameObject.Find("CenterCamera").GetComponent<Camera>();
        SafetyText.text = "Okie";
        SafetyText.color = Color.magenta;
        TTCText.text = "5.2";
        SteeringAngleText.text = "";
        SpeedText.text = "";
        DataCaptureText.text = "";
        SetAngleValue(0);
        SetMPHValue(0);
    }

    void Update()
    {
        if (carController.getSaveStatus())
        {
            DataCaptureText.text = "Capturing Data: " + (int)(100 * carController.getSavePercent()) + "%";
            //Debug.Log ("save percent is: " + carController.getSavePercent ());
        }

        SetCameraImage();

        // for photo-shooting
        SetAngleValue(0.05f);
        //SetAngleValue(carController.CurrentSteerAngle);

        SetMPHValue(carController.CurrentSpeed);


        
    }

    public void SetCameraImage()
    {
        //CenterCamera.Render();
        RenderTexture targetTexture = CenterCamera.targetTexture;
        RenderTexture.active = targetTexture;
        Texture2D tex = new Texture2D(targetTexture.width, targetTexture.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, targetTexture.width, targetTexture.height), 0, 0);
        tex.Apply();
        //tex = ConvertToGrayscale(tex);
        CameraImage.texture = tex;
    }


    public void SetAngleValue(float value)
    {
        //SteeringAngleText.text = (value*25).ToString("N2") + "°";
        SteeringAngleText.text = (value * 25).ToString("N2");
    }

    public void SetMPHValue(float value)
    {
        SpeedText.text = value.ToString("N2");
        //SpeedAnimation.fillAmount = value / carController.MaxSpeed;
    }

    Texture2D ConvertToGrayscale(Texture2D tex)
    {
        Color32[] pixels = tex.GetPixels32();
        for (int x = 0; x < tex.width; x++)
        {
            for (int y = 0; y < tex.height; y++)
            {
                Color32 pixel = pixels[x + y * tex.width];
                int p = ((256 * 256 + pixel.r) * 256 + pixel.b) * 256 + pixel.g;
                int b = p % 256;
                p = Mathf.FloorToInt(p / 256);
                int g = p % 256;
                p = Mathf.FloorToInt(p / 256);
                int r = p % 256;
                float l = (0.2126f * r / 255f) + 0.7152f * (g / 255f) + 0.0722f * (b / 255f);
                Color c = new Color(l, l, l, 1);
                tex.SetPixel(x, y, c);
            }
        }
        tex.Apply(false);

        return tex;
    }
}
