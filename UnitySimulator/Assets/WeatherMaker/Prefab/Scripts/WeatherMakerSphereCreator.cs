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

using System;
using System.Collections.Generic;

using UnityEngine;

namespace DigitalRuby.WeatherMaker
{
    public enum UVMode
    {
        /// <summary>
        /// Map the image to the full 360 degree sphere
        /// </summary>
        Sphere,

        /// <summary>
        /// Map the panorama to the top of the sphere
        /// </summary>
        Panorama,

        /// <summary>
        /// Map the panorama to the top of the sphere and mirror to the bottom half of the sphere
        /// </summary>
        PanoramaMirrorDown,

        /// <summary>
        /// Map the image to the top half of the sphere
        /// </summary>
        Dome,

        /// <summary>
        /// Map the image to the top half of the sphere and mirror to the bottom half of the sphere
        /// </summary>
        DomeMirrorDown,

        /// <summary>
        /// Dome image contains the bottom on the left hand side of the texture and top on the right hand side in 2 to 1 aspect, i.e. 4096x2048.
        /// </summary>
        DomeDouble,

        /// <summary>
        /// Map the fish eye to the sphere, mirroring to the other side of the sphere
        /// </summary>
        FishEyeMirrored,

        /// <summary>
        /// Map the fish eye to the entire sphere
        /// </summary>
        FishEye360
    }

    /// <summary>
    /// Allows creating mesh spheres
    /// </summary>
    public static class WeatherMakerSphereCreator
    {
        #region Structs

        private static readonly Vector3[] directions =
        {
            Vector3.left,
            Vector3.back,
            Vector3.right,
            Vector3.forward
        };

        private struct SharedTriangle
        {
            public int Index;
            public int A;
            public bool AShared;
            public int B;
            public bool BShared;
            public int C;
            public bool CShared;
        }

        private struct MeshData
        {
            public Mesh Mesh;
            public List<int> Triangles;
            public List<Vector3> Vertices;
            public List<Vector3> Normals;
            public List<Vector4> Tangents;
            public List<Vector2> UV;

            public void ChangeUVX(int index, float x)
            {
                Vector2 uv = UV[index];
                uv.x = x;
                UV[index] = uv;
            }

            public void Apply()
            {
                Mesh.SetVertices(Vertices);
                Mesh.SetNormals(Normals);
                Mesh.SetTangents(Tangents);
                Mesh.SetUVs(0, UV);
                Mesh.SetTriangles(Triangles, 0);
                Mesh.RecalculateBounds();
            }
        }

        #endregion Structs

        public static Mesh Create(string name, int resolution, UVMode uvMode)
        {
            if (resolution < 0)
            {
                resolution = 0;
                Debug.LogWarning("Sphere subdivisions increased to minimum, which is 0.");
            }
            else if (resolution > 6)
            {
                resolution = 6;
                Debug.LogWarning("Sphere subdivisions decreased to maximum, which is 6.");
            }

            int trianglesCapacity = (1 << (resolution * 2 + 3)) * 3;
            resolution = 1 << resolution;
            int verticesCapacity = (resolution + 1) * (resolution + 1) * 4 - (resolution * 2 - 1) * 3;
            MeshData m = CreateMesh(name, verticesCapacity, trianglesCapacity, uvMode.ToString());
            CreateInternal(ref m, resolution, uvMode);

            return m.Mesh;
        }

        private static MeshData CreateMesh(string name, int verticesCapacity, int trianglesCapacity, string meshNameSuffix)
        {
            MeshData m = new MeshData
            {
                Mesh = new Mesh { name = name + meshNameSuffix },
                Triangles = new List<int>(trianglesCapacity),
                Vertices = new List<Vector3>(verticesCapacity),
                Normals = new List<Vector3>(verticesCapacity),
                Tangents = new List<Vector4>(verticesCapacity),
                UV = new List<Vector2>(verticesCapacity)
            };

            for (int i = 0; i < verticesCapacity; i++)
            {
                m.Vertices.Add(Vector3.zero);
                m.Normals.Add(Vector3.zero);
                m.Tangents.Add(Vector4.zero);
                m.UV.Add(Vector2.zero);
            }
            for (int i = 0; i < trianglesCapacity; i++)
            {
                m.Triangles.Add(0);
            }

            return m;
        }

        private static void FixTriangle(ref Vector3 vertice, ref Vector3 p1, ref Vector3 p2, int triangleIndex, ref MeshData m)
        {
            float smallFloat = 0.0001f;
            Vector3 dir = (((p1 - vertice) + (p2 - vertice)) * 0.5f);
            float xAdd = (dir.x == 0.0f ? 0.0f : (smallFloat * Mathf.Sign(dir.x)));
            float yAdd = (dir.y == 0.0f ? 0.0f : (smallFloat * Mathf.Sign(dir.y)));
            float zAdd = (dir.z == 0.0f ? 0.0f : (smallFloat * Mathf.Sign(dir.z)));
            Vector3 norm = new Vector3(vertice.x + xAdd, vertice.y + yAdd, vertice.z + zAdd);
            Vector4 tangent = m.Tangents[m.Triangles[triangleIndex]];
            m.Triangles[triangleIndex] = m.Vertices.Count;
            m.Vertices.Add(vertice);
            m.Normals.Add(norm);
            m.Tangents.Add(tangent);
            m.UV.Add(Vector2.zero); // uv will be calculated later
        }

        private static void FixTriangle(ref Vector3 point, List<SharedTriangle> triangleList, ref MeshData m)
        {
            foreach (SharedTriangle t in triangleList)
            {
                Vector3 av = m.Vertices[t.A];
                Vector3 bv = m.Vertices[t.B];
                Vector3 cv = m.Vertices[t.C];

                if (t.AShared)
                {
                    FixTriangle(ref av, ref bv, ref cv, t.Index, ref m);
                }
                if (t.BShared)
                {
                    FixTriangle(ref bv, ref av, ref cv, t.Index + 1, ref m);
                }
                if (t.CShared)
                {
                    FixTriangle(ref cv, ref av, ref bv, t.Index + 2, ref m);
                }
            }
        }

        private static void AddTriangleIfNeeded(Dictionary<Vector3, List<SharedTriangle>> triangleList, ref Vector3 v, ref SharedTriangle t, ref bool shareVar, ref int counter)
        {
            if (v.x == 0.0f || v.y == 0.0f || v.z == 0.0f)
            {
                List<SharedTriangle> list;
                if (!triangleList.TryGetValue(v, out list))
                {
                    list = new List<SharedTriangle>();
                    triangleList[v] = list;
                }
                shareVar = true;
                list.Add(t);
                counter++;
            }
        }

        private static void FixVertices(ref MeshData m)
        {
            Dictionary<Vector3, List<SharedTriangle>> triangleList = new Dictionary<Vector3, List<SharedTriangle>>();
            int counter = 0;

            // loop through all triangles and create a list of all triangles that share a vertice that has an x, y or z value of 0
            for (int i = 0; i < m.Triangles.Count; )
            {
                SharedTriangle t = new SharedTriangle { Index = i, A = m.Triangles[i++], B = m.Triangles[i++], C = m.Triangles[i++] };
                Vector3 av = m.Vertices[t.A];
                Vector3 bv = m.Vertices[t.B];
                Vector3 cv = m.Vertices[t.C];
                AddTriangleIfNeeded(triangleList, ref av, ref t, ref t.AShared, ref counter);
                AddTriangleIfNeeded(triangleList, ref bv, ref t, ref t.BShared, ref counter);
                AddTriangleIfNeeded(triangleList, ref cv, ref t, ref t.CShared, ref counter);
            }

            foreach (KeyValuePair<Vector3, List<SharedTriangle>> kv in triangleList)
            {
                Vector3 v = kv.Key;
                FixTriangle(ref v, kv.Value, ref m);
            }
        }

        private static void CreateInternal(ref MeshData m, int resolution, UVMode uvMode)
        {
            int v = 0;
            int vBottom = 0;
            int t = 0;

            for (int i = 0; i < 4; i++)
            {
                m.Vertices[v++] = Vector3.down;
            }

            for (int i = 1; i <= resolution; i++)
            {
                float progress = (float)i / resolution;
                Vector3 from, to;
                m.Vertices[v++] = to = Vector3.Lerp(Vector3.down, Vector3.forward, progress);
                for (int d = 0; d < 4; d++)
                {
                    from = to;
                    to = Vector3.Lerp(Vector3.down, directions[d], progress);
                    t = CreateLowerStrip(i, v, vBottom, t, ref m);
                    v = CreateVertexLine(from, to, i, v, ref m);
                    vBottom += i > 1 ? (i - 1) : 1;
                }
                vBottom = v - 1 - i * 4;
            }

            for (int i = resolution - 1; i >= 1; i--)
            {
                float progress = (float)i / resolution;
                Vector3 from, to;
                m.Vertices[v++] = to = Vector3.Lerp(Vector3.up, Vector3.forward, progress);
                for (int d = 0; d < 4; d++)
                {
                    from = to;
                    to = Vector3.Lerp(Vector3.up, directions[d], progress);
                    t = CreateUpperStrip(i, v, vBottom, t, ref m);
                    v = CreateVertexLine(from, to, i, v, ref m);
                    vBottom += i + 1;
                }
                vBottom = v - 1 - i * 4;
            }

            for (int i = 0; i < 4; i++)
            {
                m.Triangles[t++] = vBottom;
                m.Triangles[t++] = v;
                m.Triangles[t++] = ++vBottom;
                m.Vertices[v++] = Vector3.up;
            }

            CreateNormals(ref m);
            CreateTangents(ref m);
            FixVertices(ref m); // must be done before CreateUV is called
            CreateUV(ref m, uvMode);

            m.Apply();
        }

        private static int CreateVertexLine(Vector3 from, Vector3 to, int steps, int v, ref MeshData m)
        {
            for (int i = 1; i <= steps; i++)
            {
                m.Vertices[v++] = Vector3.Lerp(from, to, (float)i / steps);
            }
            return v;
        }

        private static int CreateLowerStrip(int steps, int vTop, int vBottom, int t, ref MeshData m)
        {
            for (int i = 1; i < steps; i++)
            {
                m.Triangles[t++] = vBottom;
                m.Triangles[t++] = vTop - 1;
                m.Triangles[t++] = vTop;

                m.Triangles[t++] = vBottom++;
                m.Triangles[t++] = vTop++;
                m.Triangles[t++] = vBottom;
            }
            m.Triangles[t++] = vBottom;
            m.Triangles[t++] = vTop - 1;
            m.Triangles[t++] = vTop;
            return t;
        }

        private static int CreateUpperStrip(int steps, int vTop, int vBottom, int t, ref MeshData m)
        {
            m.Triangles[t++] = vBottom;
            m.Triangles[t++] = vTop - 1;
            m.Triangles[t++] = ++vBottom;
            for (int i = 1; i <= steps; i++)
            {
                m.Triangles[t++] = vTop - 1;
                m.Triangles[t++] = vTop;
                m.Triangles[t++] = vBottom;

                m.Triangles[t++] = vBottom;
                m.Triangles[t++] = vTop++;
                m.Triangles[t++] = ++vBottom;
            }
            return t;
        }

        private static void CreateNormals(ref MeshData m)
        {
            for (int i = 0; i < m.Vertices.Count; i++)
            {
                m.Normals[i] = m.Vertices[i] = m.Vertices[i].normalized;
            }
        }

        private static float WrapFloat(float v)
        {
            if (v < 0.0f)
            {
                v += 1.0f;
            }
            else if (v > 1.0f)
            {
                v -= 1.0f;
            }

            return v;
        }

        private static void CreateUV(ref MeshData m, UVMode uvMode)
        {
            for (int i = 0; i < m.UV.Count; i++)
            {
                Vector3 n = m.Normals[i];
                Vector2 textureCoordinates = new Vector2();

                switch (uvMode)
                {
                    case UVMode.Sphere:
                    {
                        textureCoordinates.x = Mathf.Atan2(n.x, n.z) / (-2f * Mathf.PI);
                        textureCoordinates.y = Mathf.Asin(n.y) / Mathf.PI + 0.5f;
                    }
                    break;

                    case UVMode.Panorama:
                    {
                        textureCoordinates.x = Mathf.Atan2(n.x, n.z) / (-2f * Mathf.PI);
                        textureCoordinates.y = Mathf.Clamp(n.y, 0.0f, 1.0f);
                    }
                    break;

                    case UVMode.PanoramaMirrorDown:
                    {
                        textureCoordinates.x = Mathf.Atan2(n.x, n.z) / (-2f * Mathf.PI);
                        textureCoordinates.y = Mathf.Abs(n.y);
                    }
                    break;

                    case UVMode.Dome:
                    {
                        float y = (n.y < 0.0f ? 0.0f : n.y);
                        float r = Mathf.Atan2(Mathf.Sqrt(n.x * n.x + n.z * n.z), y) / Mathf.PI;
                        float phi = Mathf.Atan2(n.z, n.x);
                        textureCoordinates.x = (r * Mathf.Cos(phi)) + 0.5f;
                        textureCoordinates.y = (r * Mathf.Sin(phi)) + 0.5f;
                    }
                    break;

                    case UVMode.DomeMirrorDown:
                    {
                        float y = Math.Abs(n.y);
                        float r = Mathf.Atan2(Mathf.Sqrt(n.x * n.x + n.z * n.z), y) / Mathf.PI;
                        float phi = Mathf.Atan2(n.z, n.x);
                        textureCoordinates.x = (r * Mathf.Cos(phi)) + 0.5f;
                        textureCoordinates.y = (r * Mathf.Sin(phi)) + 0.5f;
                    }
                    break;

                    case UVMode.DomeDouble:
                    {
                        float y = Math.Abs(n.y);
                        float r = Mathf.Atan2(Mathf.Sqrt(n.x * n.x + n.z * n.z), y) / Mathf.PI;
                        float phi = Mathf.Atan2(n.z, n.x);
                        textureCoordinates.x = (r * Mathf.Cos(phi)) + 0.5f;
                        textureCoordinates.y = (r * Mathf.Sin(phi)) + 0.5f;
                        if (n.y >= 0.0f)
                        {
                            textureCoordinates.x = (textureCoordinates.x + 1.0f) * 0.5f;
                        }
                        else
                        {
                            textureCoordinates.x *= 0.5f;
                        }
                    }
                    break;

                    case UVMode.FishEyeMirrored:
                    {
                        float z = Math.Abs(n.z);
                        float r = Mathf.Atan2(Mathf.Sqrt(n.x * n.x + n.y * n.y), z) / Mathf.PI;
                        float phi = Mathf.Atan2(n.y, n.x);
                        textureCoordinates.x = (r * Mathf.Cos(phi)) + 0.5f;
                        textureCoordinates.y = (r * Mathf.Sin(phi)) + 0.5f;
                    }
                    break;

                    case UVMode.FishEye360:
                    {
                        float r = Mathf.Atan2(Mathf.Sqrt(n.x * n.x + n.y * n.y), n.z) / (Mathf.PI * 2.0f);

                        // TODO: The fish eye gets distorted at these coordinates, not sure if this is fixable or not...
                        /*
                        const float t = 0.35f;
                        if (n.x >= -t && n.x <= t && n.y >= -t && n.y <= t)
                        {
                            //Debug.LogFormat("BOO: n={0}, t={1}, r={2}, phi={3}", n, textureCoordinates, r, phi);
                            //textureCoordinates.x = 0.0f;
                            //textureCoordinates.y = 0.0f;
                        }
                        */

                        float phi = Mathf.Atan2(n.y, n.x);
                        textureCoordinates.x = (r * Mathf.Cos(phi)) + 0.5f;
                        textureCoordinates.y = (r * Mathf.Sin(phi)) + 0.5f;
                    }
                    break;

                    default: throw new ArgumentException("Invalid uv mode: " + uvMode);
                }

                textureCoordinates.x = WrapFloat(textureCoordinates.x);
                textureCoordinates.y = WrapFloat(textureCoordinates.y);
                m.UV[i] = textureCoordinates;
            }
        }

        private static void CreateTangents(ref MeshData m)
        {
            for (int i = 0; i < m.Vertices.Count; i++)
            {
                Vector3 v = m.Vertices[i];
                v.y = 0.0f;
                v = v.normalized;
                Vector4 tangent;
                tangent.x = -v.z;
                tangent.y = 0.0f;
                tangent.z = v.x;
                tangent.w = -1.0f;
                m.Tangents[i] = tangent;
            }

            m.Tangents[m.Tangents.Count - 4] = m.Tangents[0] = new Vector3(-1f, 0, -1f).normalized;
            m.Tangents[m.Tangents.Count - 3] = m.Tangents[1] = new Vector3(1f, 0f, -1f).normalized;
            m.Tangents[m.Tangents.Count - 2] = m.Tangents[2] = new Vector3(1f, 0f, 1f).normalized;
            m.Tangents[m.Tangents.Count - 1] = m.Tangents[3] = new Vector3(-1f, 0f, 1f).normalized;
            for (int i = 0; i < 4; i++)
            {
                Vector4 t1 = m.Tangents[m.Tangents.Count - 1 - 1];
                Vector4 t2 = m.Tangents[i];
                t1.w = t2.w = -1.0f;
                m.Tangents[m.Tangents.Count - 1 - 1] = t1;
                m.Tangents[i] = t2;
            }
        }
    }
}