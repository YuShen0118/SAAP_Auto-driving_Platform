using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    public class WeatherMakerPlayerControllerScript : MonoBehaviour
    {
        [Range(0.0f, 5000.0f)]
        public float MovementSpeed = 20.0f;
        public bool EnableMouseLook;
        public Light Flashlight;

        private enum RotationAxes { MouseXAndY = 0, MouseX = 1, MouseY = 2 }
        private RotationAxes axes = RotationAxes.MouseXAndY;
        private float sensitivityX = 15F;
        private float sensitivityY = 15F;
        private float minimumX = -360F;
        private float maximumX = 360F;
        private float minimumY = -60F;
        private float maximumY = 60F;
        private float rotationX = 0F;
        private float rotationY = 0F;
        private Quaternion originalRotation;

        private void UpdateMouseLook()
        {
            if (Input.GetKeyDown(KeyCode.M))
            {
                EnableMouseLook = !EnableMouseLook;
                GameObject toggleObj = GameObject.Find("MouseLookEnabledCheckBox");
                if (toggleObj != null)
                {
                    UnityEngine.UI.Toggle movementToggle = toggleObj.GetComponent<UnityEngine.UI.Toggle>();
                    if (movementToggle != null)
                    {
                        movementToggle.isOn = EnableMouseLook;
                    }
                }
            }
            if (!EnableMouseLook || MovementSpeed <= 0.0f)
            {
                return;
            }
            if (axes == RotationAxes.MouseXAndY)
            {
                // Read the mouse input axis
                rotationX += Input.GetAxis("Mouse X") * sensitivityX;
                rotationY += Input.GetAxis("Mouse Y") * sensitivityY;

                rotationX = ClampAngle(rotationX, minimumX, maximumX);
                rotationY = ClampAngle(rotationY, minimumY, maximumY);

                Quaternion xQuaternion = Quaternion.AngleAxis(rotationX, Vector3.up);
                Quaternion yQuaternion = Quaternion.AngleAxis(rotationY, -Vector3.right);

                transform.localRotation = originalRotation * xQuaternion * yQuaternion;
            }
            else if (axes == RotationAxes.MouseX)
            {
                rotationX += Input.GetAxis("Mouse X") * sensitivityX;
                rotationX = ClampAngle(rotationX, minimumX, maximumX);

                Quaternion xQuaternion = Quaternion.AngleAxis(rotationX, Vector3.up);
                transform.localRotation = originalRotation * xQuaternion;
            }
            else
            {
                rotationY += Input.GetAxis("Mouse Y") * sensitivityY;
                rotationY = ClampAngle(rotationY, minimumY, maximumY);

                Quaternion yQuaternion = Quaternion.AngleAxis(-rotationY, Vector3.right);
                transform.localRotation = originalRotation * yQuaternion;
            }
        }

        private void UpdateMovement()
        {
            if (MovementSpeed <= 0.0f)
            {
                return;
            }

            float speed = MovementSpeed * Time.deltaTime;

            if (Input.GetKey(KeyCode.W))
            {
                transform.Translate(0.0f, 0.0f, speed);
            }
            if (Input.GetKey(KeyCode.S))
            {
                transform.Translate(0.0f, 0.0f, -speed);
            }
            if (Input.GetKey(KeyCode.A))
            {
                transform.Translate(-speed, 0.0f, 0.0f);
            }
            if (Input.GetKey(KeyCode.D))
            {
                transform.Translate(speed, 0.0f, 0.0f);
            }
        }

        private void Start()
        {
            originalRotation = transform.localRotation;
        }

        private void Update()
        {
            UpdateMovement();
            UpdateMouseLook();
            if (Flashlight != null)
            {
                if (Input.GetKeyDown(KeyCode.F))
                {
                    // hack: Bug in Unity, doesn't recognize that the light was enabled unless we rotate the camera
                    transform.Rotate(0.0f, 0.01f, 0.0f);
                    transform.Rotate(0.0f, -0.01f, 0.0f);

                    Flashlight.enabled = !Flashlight.enabled;
                    GameObject toggleObj = GameObject.Find("FlashlightCheckbox");
                    if (toggleObj != null)
                    {
                        UnityEngine.UI.Toggle toggle = toggleObj.GetComponent<UnityEngine.UI.Toggle>();
                        if (toggle != null)
                        {
                            toggle.isOn = !toggle.isOn;
                        }
                    }
                }
                WeatherMakerLightManagerScript.Instance.AddLight(Flashlight);
            }
        }

        public static float ClampAngle(float angle, float min, float max)
        {
            if (angle < -360F)
            {
                angle += 360F;
            }
            if (angle > 360F)
            {
                angle -= 360F;
            }

            return Mathf.Clamp(angle, min, max);
        }
    }
}
