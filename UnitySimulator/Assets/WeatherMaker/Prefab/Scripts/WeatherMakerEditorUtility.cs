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

#if UNITY_EDITOR

using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;

#endif

using System.Collections.Generic;
using System.IO;

namespace DigitalRuby.WeatherMaker
{
    [System.Serializable]
    public struct RangeOfIntegers
    {
        [Tooltip("Minimum value (inclusive)")]
        public int Minimum;

        [Tooltip("Maximum value (inclusive)")]
        public int Maximum;

        public int Random() { return UnityEngine.Random.Range(Minimum, Maximum + 1); }
        public int Random(System.Random r) { return r.Next(Minimum, Maximum + 1); }
    }

    [System.Serializable]
    public struct RangeOfFloats
    {
        [Tooltip("Minimum value (inclusive)")]
        public float Minimum;

        [Tooltip("Maximum value (inclusive)")]
        public float Maximum;

        public float Random() { return UnityEngine.Random.Range(Minimum, Maximum); }
        public float Random(System.Random r) { return Minimum + ((float)r.NextDouble() * (Maximum - Minimum)); }
    }

    public class SingleLineAttribute : PropertyAttribute
    {
        public SingleLineAttribute(string tooltip) { Tooltip = tooltip; }

        public string Tooltip { get; private set; }
    }

    public class SingleLineClampAttribute : SingleLineAttribute
    {
        public SingleLineClampAttribute(string tooltip, float minValue, float maxValue) : base(tooltip)
        {
            MinValue = minValue;
            MaxValue = maxValue;
        }

        public float MinValue { get; private set; }
        public float MaxValue { get; private set; }
    }

    /// <summary>
    /// Helper methods when serializing / deserializing fields with dynamic editor scripts
    /// </summary>
    public static class SerializationHelper
    {
        public const byte HeaderFloat = 0;
        public const byte HeaderInt = 1;
        public const byte HeaderShort = 2;
        public const byte HeaderByte = 3;
        public const byte HeaderColor = 4;
        public const byte HeaderVector2 = 5;
        public const byte HeaderVector3 = 6;
        public const byte HeaderVector4 = 7;
        public const byte HeaderQuaternion = 8;
        public const byte HeaderEnum = 9;
        public const byte HeaderBool = 10;
        public const byte HeaderFloatRange = 11;
        public const byte HeaderIntRange = 12;

        public static readonly System.Collections.Generic.Dictionary<System.Type, byte> TypesToHeader = new System.Collections.Generic.Dictionary<System.Type, byte>
        {
            { typeof(float), HeaderFloat },
            { typeof(int), HeaderInt },
            { typeof(short), HeaderShort },
            { typeof(byte), HeaderByte },
            { typeof(Color), HeaderColor },
            { typeof(Vector2), HeaderVector2 },
            { typeof(Vector3), HeaderVector3 },
            { typeof(Vector4), HeaderVector4 },
            { typeof(Quaternion), HeaderQuaternion },
            { typeof(System.Enum), HeaderEnum },
            { typeof(bool), HeaderBool },
            { typeof(RangeOfFloats), HeaderFloatRange },
            { typeof(RangeOfIntegers), HeaderIntRange }
        };


        public static byte[] Serialize(object obj)
        {
            if (obj == null)
            {
                return null;
            }
            MemoryStream ms = new MemoryStream();
            BinaryWriter writer = new BinaryWriter(ms, System.Text.Encoding.UTF8);
            System.Type t = obj.GetType();
            byte header;
            if (!TypesToHeader.TryGetValue(t, out header))
            {

#if NETFX_CORE

                if (!System.Reflection.IntrospectionExtensions.GetTypeInfo(t).IsEnum)

#else

                if (!t.IsEnum)

#endif

                {
                    return null;
                }
                header = HeaderEnum;
            }
            writer.Write(header);
            switch (header)
            {
                case HeaderFloat: { writer.Write((float)obj); break; }
                case HeaderInt: { writer.Write((int)obj); break; }
                case HeaderShort: { writer.Write((short)obj); break; }
                case HeaderByte: { writer.Write((byte)obj); break; }
                case HeaderColor: { Color c = (Color)obj; writer.Write(c.r); writer.Write(c.g); writer.Write(c.b); writer.Write(c.a); break; }
                case HeaderVector2: { Vector2 v = (Vector2)obj; writer.Write(v.x); writer.Write(v.y); break; }
                case HeaderVector3: { Vector3 v = (Vector3)obj; writer.Write(v.x); writer.Write(v.y); writer.Write(v.z); break; }
                case HeaderVector4: { Vector4 v = (Vector4)obj; writer.Write(v.x); writer.Write(v.y); writer.Write(v.z); writer.Write(v.w); break; }
                case HeaderQuaternion: { Quaternion q = (Quaternion)obj; writer.Write(q.x); writer.Write(q.y); writer.Write(q.z); writer.Write(q.w); break; }
                case HeaderBool: { writer.Write((bool)obj); break; }
                case HeaderFloatRange: { RangeOfFloats v = (RangeOfFloats)obj; writer.Write(v.Minimum); writer.Write(v.Maximum); break; }
                case HeaderIntRange: { RangeOfIntegers v = (RangeOfIntegers)obj; writer.Write(v.Minimum); writer.Write(v.Maximum); break; }
                case HeaderEnum: { writer.Write((int)obj); break; }
            }
            return ms.ToArray();
        }

        public static object Deserialize(byte[] bytes, System.Type type = null)
        {
            if (bytes == null || bytes.Length < 2)
            {
                return null;
            }
            MemoryStream ms = new MemoryStream(bytes);
            BinaryReader reader = new BinaryReader(ms, System.Text.Encoding.UTF8);
            switch (reader.ReadByte())
            {
                case HeaderFloat: return reader.ReadSingle();
                case HeaderInt: return reader.ReadInt32();
                case HeaderShort: return reader.ReadInt16();
                case HeaderByte: return reader.ReadByte();
                case HeaderColor: return new Color(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                case HeaderVector2: return new Vector2(reader.ReadSingle(), reader.ReadSingle());
                case HeaderVector3: return new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                case HeaderVector4: return new Vector4(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                case HeaderQuaternion: return new Quaternion(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                case HeaderBool: return reader.ReadBoolean();
                case HeaderFloatRange: return new RangeOfFloats { Minimum = reader.ReadSingle(), Maximum = reader.ReadSingle() };
                case HeaderIntRange: return new RangeOfIntegers { Minimum = reader.ReadInt32(), Maximum = reader.ReadInt32() };
                case HeaderEnum: return (type == null ? reader.ReadInt32() : System.Enum.ToObject(type, reader.ReadInt32()));
                default: return null;
            }
        }

#if UNITY_EDITOR

        public static void RepaintInspector()
        {
            Editor[] ed = (Editor[])Resources.FindObjectsOfTypeAll<Editor>();
            for (int i = 0; i < ed.Length; i++)
            {
                ed[i].Repaint();
            }
        }

        /// <summary>
        /// Mark an object as dirty - only works in editor mode
        /// </summary>
        /// <param name="obj">Object to set dirty</param>
        public static void SetDirty(UnityEngine.Object obj)
        {
            if (Application.isPlaying || obj == null)
            {
                return;
            }
            Undo.RecordObject(obj, "WeatherMaker Change");
            EditorUtility.SetDirty(obj);
            if (!EditorUtility.IsPersistent(obj))
            {
                MonoBehaviour mb = obj as MonoBehaviour;
                if (mb != null)
                {
                    EditorSceneManager.MarkSceneDirty(mb.gameObject.scene);
                    return;
                }
                GameObject go = obj as GameObject;
                if (go != null)
                {
                    EditorSceneManager.MarkSceneDirty(go.scene);
                    return;
                }
                EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
            }
        }

#endif

    }

    public class EnumFlagAttribute : PropertyAttribute { }

    public class ReadOnlyLabelAttribute : PropertyAttribute { }

    public enum HelpBoxMessageType { None, Info, Warning, Error }

    public class HelpBoxAttribute : PropertyAttribute
    {
        public string Text { get; private set; }
        public HelpBoxMessageType MessageType { get; private set; }

        public HelpBoxAttribute(string text, HelpBoxMessageType messageType = HelpBoxMessageType.None)
        {
            Text = text;
            MessageType = messageType;
        }
    }

#if UNITY_EDITOR

    [CustomPropertyDrawer(typeof(SingleLineAttribute))]
    [CustomPropertyDrawer(typeof(SingleLineClampAttribute))]
    public class SingleLineDrawer : PropertyDrawer
    {
        public override void OnGUI(Rect position, SerializedProperty prop, GUIContent label)
        {
            string tooltip = (attribute as SingleLineAttribute).Tooltip;
            label.tooltip = tooltip;
            EditorGUI.BeginProperty(position, label, prop);
            position = EditorGUI.PrefixLabel(position, GUIUtility.GetControlID(FocusType.Passive), label);
            int indent = EditorGUI.indentLevel;
            EditorGUI.indentLevel = 0;
            switch (prop.type)
            {
                case "RangeOfIntegers":
                    PropertyDrawerExtensions.DrawRangeField(this, position, prop, false);
                    break;

                case "RangeOfFloats":
                    PropertyDrawerExtensions.DrawRangeField(this, position, prop, true);
                    break;

                default:
                    EditorGUI.HelpBox(position, "[SingleLineDrawer] doesn't work with type '" + prop.type + "'", MessageType.Error);
                    break;
            }
            EditorGUI.indentLevel = indent;
            EditorGUI.EndProperty();
        }
    }

    [CustomPropertyDrawer(typeof(HelpBoxAttribute))]
    public class HelpBoxAttributeDrawer : DecoratorDrawer
    {
        public override float GetHeight()
        {
            var helpBoxAttribute = attribute as HelpBoxAttribute;
            if (helpBoxAttribute == null) return base.GetHeight();
            var helpBoxStyle = (GUI.skin != null) ? GUI.skin.GetStyle("helpbox") : null;
            if (helpBoxStyle == null) return base.GetHeight();
            return helpBoxStyle.CalcHeight(new GUIContent(helpBoxAttribute.Text), EditorGUIUtility.currentViewWidth) + 4.0f;
        }

        public override void OnGUI(Rect position)
        {
            var helpBoxAttribute = attribute as HelpBoxAttribute;
            if (helpBoxAttribute == null) return;
            EditorGUI.HelpBox(position, helpBoxAttribute.Text, GetMessageType(helpBoxAttribute.MessageType));
        }

        private MessageType GetMessageType(HelpBoxMessageType helpBoxMessageType)
        {
            switch (helpBoxMessageType)
            {
                default:
                case HelpBoxMessageType.None: return MessageType.None;
                case HelpBoxMessageType.Info: return MessageType.Info;
                case HelpBoxMessageType.Warning: return MessageType.Warning;
                case HelpBoxMessageType.Error: return MessageType.Error;
            }
        }
    }

    public static class PropertyDrawerExtensions
    {
        public static void DrawIntTextField(this PropertyDrawer drawer, Rect position, string text, string tooltip, SerializedProperty prop)
        {
            EditorGUI.BeginChangeCheck();
            int value = EditorGUI.IntField(position, new GUIContent(text, tooltip), prop.intValue);
            SingleLineClampAttribute clamp = drawer.attribute as SingleLineClampAttribute;
            if (clamp != null)
            {
                value = Mathf.Clamp(value, (int)clamp.MinValue, (int)clamp.MaxValue);
            }
            if (EditorGUI.EndChangeCheck())
            {
                prop.intValue = value;
            }
        }

        public static void DrawFloatTextField(this PropertyDrawer drawer, Rect position, string text, string tooltip, SerializedProperty prop)
        {
            EditorGUI.BeginChangeCheck();
            float value = EditorGUI.FloatField(position, new GUIContent(text, tooltip), prop.floatValue);
            SingleLineClampAttribute clamp = drawer.attribute as SingleLineClampAttribute;
            if (clamp != null)
            {
                value = Mathf.Clamp(value, (float)clamp.MinValue, (float)clamp.MaxValue);
            }
            if (EditorGUI.EndChangeCheck())
            {
                prop.floatValue = value;
            }
        }

        public static void DrawRangeField(this PropertyDrawer drawer, Rect position, SerializedProperty prop, bool floatingPoint)
        {
            EditorGUIUtility.labelWidth = 30.0f;
            EditorGUIUtility.fieldWidth = 60.0f;
            float width = 100.0f;
            float spacing = 10.0f;
            position.x -= (EditorGUI.indentLevel * 15.0f);
            position.width = width;
            if (floatingPoint)
            {
                DrawFloatTextField(drawer, position, "Min", "Minimum value", prop.FindPropertyRelative("Minimum"));
            }
            else
            {
                DrawIntTextField(drawer, position, "Min", "Minimum value", prop.FindPropertyRelative("Minimum"));
            }
            position.x = position.xMax + spacing;
            position.width = width;
            if (floatingPoint)
            {
                DrawFloatTextField(drawer, position, "Max", "Maximum value", prop.FindPropertyRelative("Maximum"));
            }
            else
            {
                DrawIntTextField(drawer, position, "Max", "Maximum value", prop.FindPropertyRelative("Maximum"));
            }
        }
    }

    public class PopupList : PopupWindowContent
    {
        private Vector2 scrollViewOffset;
        private static Texture2D gray;

        public Vector2 Size = new Vector2(300.0f, 200.0f);
        public string Title = string.Empty;
        public GUIStyle ListStyle { get; set; }
        public GUIContent[] Items { get; set; }
        public int SelectedItemIndex { get; set; }
        public bool IsShowing { get; private set; }

        public PopupList()
        {
            ListStyle = "label";
            if (gray == null)
            {
                gray = new Texture2D(1, 1);
                gray.SetPixel(0, 0, Color.gray);
                gray.Apply();
            }
        }

        public override Vector2 GetWindowSize()
        {
            return Size;
        }

        public override void OnGUI(Rect rect)
        {
            if (!string.IsNullOrEmpty(Title))
            {
                GUILayout.Label(Title, EditorStyles.boldLabel);
            }
            scrollViewOffset = EditorGUILayout.BeginScrollView(scrollViewOffset);
            Texture2D origBg = ListStyle.normal.background;
            for (int i = 0; i < Items.Length; i++)
            {
                GUIStyle style = ListStyle;
                Rect r = GUILayoutUtility.GetRect(Items[i], style);
                if (i == SelectedItemIndex || r.Contains(Event.current.mousePosition))
                {
                    style = new GUIStyle(style);
                    style.normal.textColor = style.hover.textColor = Color.white;
                    style.normal.background = style.hover.background = gray;
                }
                if (GUI.Button(r, Items[i], style))
                {
                    SelectedItemIndex = i;
                    if (SelectedIndexChanged != null)
                    {
                        SelectedIndexChanged(this, System.EventArgs.Empty);
                    }
                    editorWindow.Close();
                }
            }
            ListStyle.normal.background = origBg;
            GUI.EndScrollView();
        }

        public override void OnOpen()
        {
            IsShowing = true;
            scrollViewOffset.y = ListStyle.CalcHeight(Items[0], 1.0f) * SelectedItemIndex;
        }

        public override void OnClose()
        {
            IsShowing = false;
        }

        public System.EventHandler SelectedIndexChanged;
    }

    public class EditorDrawLine
    {
        //****************************************************************************************************
        //  static function DrawLine(rect : Rect) : void
        //  static function DrawLine(rect : Rect, color : Color) : void
        //  static function DrawLine(rect : Rect, width : float) : void
        //  static function DrawLine(rect : Rect, color : Color, width : float) : void
        //  static function DrawLine(Vector2 pointA, Vector2 pointB) : void
        //  static function DrawLine(Vector2 pointA, Vector2 pointB, color : Color) : void
        //  static function DrawLine(Vector2 pointA, Vector2 pointB, width : float) : void
        //  static function DrawLine(Vector2 pointA, Vector2 pointB, color : Color, width : float) : void
        //  
        //  Draws a GUI line on the screen.
        //  
        //  DrawLine makes up for the severe lack of 2D line rendering in the Unity runtime GUI system.
        //  This function works by drawing a 1x1 texture filled with a color, which is then scaled
        //   and rotated by altering the GUI matrix.  The matrix is restored afterwards.
        //****************************************************************************************************

        public static Texture2D lineTex;

        public static void DrawLine(Rect rect) { DrawLine(rect, GUI.contentColor, 1.0f); }
        public static void DrawLine(Rect rect, Color color) { DrawLine(rect, color, 1.0f); }
        public static void DrawLine(Rect rect, float width) { DrawLine(rect, GUI.contentColor, width); }
        public static void DrawLine(Rect rect, Color color, float width) { DrawLine(new Vector2(rect.x, rect.y), new Vector2(rect.x + rect.width, rect.y + rect.height), color, width); }
        public static void DrawLine(Vector2 pointA, Vector2 pointB) { DrawLine(pointA, pointB, GUI.contentColor, 1.0f); }
        public static void DrawLine(Vector2 pointA, Vector2 pointB, Color color) { DrawLine(pointA, pointB, color, 1.0f); }
        public static void DrawLine(Vector2 pointA, Vector2 pointB, float width) { DrawLine(pointA, pointB, GUI.contentColor, width); }
        public static void DrawLine(Vector2 pointA, Vector2 pointB, Color color, float width)
        {
            // Save the current GUI matrix, since we're going to make changes to it.
            Matrix4x4 matrix = GUI.matrix;

            // Generate a single pixel texture if it doesn't exist
            if (!lineTex) { lineTex = new Texture2D(1, 1); }

            // Store current GUI color, so we can switch it back later,
            // and set the GUI color to the color parameter
            Color savedColor = GUI.color;
            GUI.color = color;

            // Determine the angle of the line.
            float angle = Vector3.Angle(pointB - pointA, Vector2.right);

            // Vector3.Angle always returns a positive number.
            // If pointB is above pointA, then angle needs to be negative.
            if (pointA.y > pointB.y) { angle = -angle; }

            // Use ScaleAroundPivot to adjust the size of the line.
            // We could do this when we draw the texture, but by scaling it here we can use
            //  non-integer values for the width and length (such as sub 1 pixel widths).
            // Note that the pivot point is at +.5 from pointA.y, this is so that the width of the line
            //  is centered on the origin at pointA.
            GUIUtility.ScaleAroundPivot(new Vector2((pointB - pointA).magnitude, width), new Vector2(pointA.x, pointA.y + 0.5f));

            // Set the rotation for the line.
            //  The angle was calculated with pointA as the origin.
            GUIUtility.RotateAroundPivot(angle, pointA);

            // Finally, draw the actual line.
            // We're really only drawing a 1x1 texture from pointA.
            // The matrix operations done with ScaleAroundPivot and RotateAroundPivot will make this
            //  render with the proper width, length, and angle.
            GUI.DrawTexture(new Rect(pointA.x, pointA.y, 1, 1), lineTex);

            // We're done.  Restore the GUI matrix and GUI color to whatever they were before.
            GUI.matrix = matrix;
            GUI.color = savedColor;
        }
    }

    [CustomPropertyDrawer(typeof(EnumFlagAttribute))]
    public class EnumFlagDrawer : PropertyDrawer
    {
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            EnumFlagAttribute flagSettings = (EnumFlagAttribute)attribute;
            System.Enum targetEnum = (System.Enum)System.Enum.ToObject(fieldInfo.FieldType, property.intValue);
            EditorGUI.BeginProperty(position, label, property);
            position = EditorGUI.PrefixLabel(position, GUIUtility.GetControlID(FocusType.Passive), label);
            System.Enum enumNew = EditorGUI.EnumMaskField(position, targetEnum);
            property.intValue = (int)System.Convert.ChangeType(enumNew, targetEnum.GetType());
            EditorGUI.EndProperty();
        }
    }

    [CustomPropertyDrawer(typeof(ReadOnlyLabelAttribute))]
    public class CommentDrawer : PropertyDrawer
    {
        public override void OnGUI(Rect position, SerializedProperty prop, GUIContent label)
        {
            EditorGUI.LabelField(position, prop.stringValue);
        }
    }

#endif

}
