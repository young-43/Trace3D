using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class Trace3DPlyVoxelMeshRenderer : MonoBehaviour
{
    [Header("PLY Input (set one)")]
    public TextAsset plyAsset;
    public string plyFilePath;

    [Header("Rendering")]
    public Material renderMaterial;
    public bool recalculateNormals = false;

    [Header("Voxel Mesh")]
    [Min(0.0001f)] public float voxelSize = 0.02f;
    [Range(0, 2)] public int dilationSteps = 1;
    public bool flipZ = true;
    [Range(1, 32)] public int pointStride = 1;
    [Min(0)] public int maxInputPoints = 600000;
    [Min(0)] public int maxVoxels = 300000;

    [Header("Run")]
    public bool autoBuildOnStart = true;
    public bool buildCollider = false;

    const float SH_C0 = 0.28209479177387814f;

    enum PlyFormat
    {
        Ascii,
        BinaryLittleEndian,
        BinaryBigEndian,
    }

    struct PlyProperty
    {
        public string type;
        public string name;
    }

    struct PropertyIndices
    {
        public int x, y, z;
        public int r, g, b;
        public int f0, f1, f2;
    }

    struct PointColor
    {
        public Vector3 pos;
        public Color color;
    }

    struct VoxelColorAcc
    {
        public int count;
        public Vector3 colorSum;

        public void Add(Color c)
        {
            count++;
            colorSum += new Vector3(c.r, c.g, c.b);
        }

        public Color Avg()
        {
            if (count <= 0) return Color.white;
            Vector3 c = colorSum / count;
            return new Color(Mathf.Clamp01(c.x), Mathf.Clamp01(c.y), Mathf.Clamp01(c.z), 1f);
        }
    }

    static readonly Vector3Int[] kNeigh6 =
    {
        new Vector3Int(1, 0, 0), new Vector3Int(-1, 0, 0),
        new Vector3Int(0, 1, 0), new Vector3Int(0, -1, 0),
        new Vector3Int(0, 0, 1), new Vector3Int(0, 0, -1),
    };

    void Start()
    {
        if (autoBuildOnStart) BuildMeshFromPly();
    }

    [ContextMenu("Build Mesh From PLY")]
    public void BuildMeshFromPly()
    {
        var t0 = Time.realtimeSinceStartup;
        try
        {
            byte[] bytes = LoadPlyBytes();
            List<PointColor> points = ReadPlyPoints(bytes);
            if (points.Count == 0)
            {
                Debug.LogError("Trace3DPlyVoxelMeshRenderer: no valid points in PLY.");
                return;
            }
            BuildVoxelMesh(points);
            float dt = Time.realtimeSinceStartup - t0;
            Debug.Log($"Trace3DPlyVoxelMeshRenderer: build finished in {dt:F2}s");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Trace3DPlyVoxelMeshRenderer: {ex.Message}");
        }
    }

    byte[] LoadPlyBytes()
    {
        if (plyAsset != null)
        {
            if (plyAsset.bytes != null && plyAsset.bytes.Length > 0) return plyAsset.bytes;
            return Encoding.UTF8.GetBytes(plyAsset.text ?? string.Empty);
        }

        if (string.IsNullOrWhiteSpace(plyFilePath))
            throw new InvalidOperationException("Please set plyAsset or plyFilePath.");
        if (!File.Exists(plyFilePath))
            throw new FileNotFoundException($"PLY file not found: {plyFilePath}");
        return File.ReadAllBytes(plyFilePath);
    }

    List<PointColor> ReadPlyPoints(byte[] data)
    {
        ParseHeader(data, out var format, out int vertexCount, out var properties, out int headerBytes);

        if (vertexCount <= 0)
            throw new InvalidDataException("Invalid PLY vertex count.");

        var indices = BuildPropertyIndices(properties);

        switch (format)
        {
            case PlyFormat.Ascii:
                return ReadAsciiVertices(data, headerBytes, vertexCount, properties, indices);
            case PlyFormat.BinaryLittleEndian:
                return ReadBinaryVertices(data, headerBytes, vertexCount, properties, indices, true);
            case PlyFormat.BinaryBigEndian:
                return ReadBinaryVertices(data, headerBytes, vertexCount, properties, indices, false);
            default:
                throw new InvalidDataException("Unsupported PLY format.");
        }
    }

    static void ParseHeader(byte[] data, out PlyFormat format, out int vertexCount, out List<PlyProperty> props, out int headerBytes)
    {
        format = PlyFormat.Ascii;
        vertexCount = 0;
        props = new List<PlyProperty>();
        headerBytes = -1;

        bool inVertexElement = false;
        int i = 0;
        while (i < data.Length)
        {
            int lineStart = i;
            while (i < data.Length && data[i] != (byte)'\n') i++;
            int lineLen = i - lineStart;
            string line = Encoding.ASCII.GetString(data, lineStart, lineLen).TrimEnd('\r');
            if (i < data.Length && data[i] == (byte)'\n') i++;

            if (line.StartsWith("format ", StringComparison.Ordinal))
            {
                if (line.Contains("ascii")) format = PlyFormat.Ascii;
                else if (line.Contains("binary_little_endian")) format = PlyFormat.BinaryLittleEndian;
                else if (line.Contains("binary_big_endian")) format = PlyFormat.BinaryBigEndian;
                else throw new InvalidDataException($"Unsupported PLY format line: {line}");
            }
            else if (line.StartsWith("element ", StringComparison.Ordinal))
            {
                string[] t = SplitWS(line);
                bool isVertexElement = t.Length >= 3 && t[1] == "vertex";
                if (isVertexElement)
                {
                    if (!int.TryParse(t[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out vertexCount))
                        throw new InvalidDataException($"Invalid vertex element line: {line}");
                }
                inVertexElement = isVertexElement;
            }
            else if (inVertexElement && line.StartsWith("property ", StringComparison.Ordinal))
            {
                string[] t = SplitWS(line);
                if (t.Length >= 3 && t[1] != "list")
                {
                    props.Add(new PlyProperty { type = t[1], name = t[2] });
                }
                else if (t.Length >= 5 && t[1] == "list")
                {
                    throw new InvalidDataException("Vertex list properties in PLY vertex element are not supported.");
                }
            }
            else if (line == "end_header")
            {
                headerBytes = i;
                break;
            }
        }

        if (headerBytes < 0) throw new InvalidDataException("PLY header missing end_header.");
        if (props.Count == 0) throw new InvalidDataException("No vertex properties found in PLY.");
    }

    static PropertyIndices BuildPropertyIndices(List<PlyProperty> props)
    {
        return new PropertyIndices
        {
            x = FindProp(props, "x"),
            y = FindProp(props, "y"),
            z = FindProp(props, "z"),
            r = FindProp(props, "red"),
            g = FindProp(props, "green"),
            b = FindProp(props, "blue"),
            f0 = FindProp(props, "f_dc_0"),
            f1 = FindProp(props, "f_dc_1"),
            f2 = FindProp(props, "f_dc_2"),
        };
    }

    bool ShouldKeepPoint(int vertexIndex, int keptCount)
    {
        if (pointStride > 1 && (vertexIndex % pointStride) != 0) return false;
        if (maxInputPoints > 0 && keptCount >= maxInputPoints) return false;
        return true;
    }

    List<PointColor> ReadAsciiVertices(byte[] data, int headerBytes, int vertexCount, List<PlyProperty> props, PropertyIndices idx)
    {
        int reserve = vertexCount / Mathf.Max(1, pointStride);
        if (maxInputPoints > 0) reserve = Mathf.Min(reserve, maxInputPoints);
        var points = new List<PointColor>(Mathf.Max(1, reserve));

        using (var ms = new MemoryStream(data, headerBytes, data.Length - headerBytes, false))
        using (var sr = new StreamReader(ms, Encoding.UTF8, true, 1024, false))
        {
            string line;
            int vertexIndex = 0;
            while (vertexIndex < vertexCount && (line = sr.ReadLine()) != null)
            {
                line = line.Trim();
                if (line.Length == 0) continue; // don't consume vertex index for empty lines

                if (!ShouldKeepPoint(vertexIndex, points.Count))
                {
                    vertexIndex++;
                    continue;
                }

                string[] t = SplitWS(line);
                if (t.Length >= props.Count)
                {
                    PointColor p;
                    if (TryExtractPoint(t, idx, out p)) points.Add(p);
                }
                vertexIndex++;
            }
        }
        return points;
    }

    List<PointColor> ReadBinaryVertices(byte[] data, int headerBytes, int vertexCount, List<PlyProperty> props, PropertyIndices idx, bool littleEndian)
    {
        int reserve = vertexCount / Mathf.Max(1, pointStride);
        if (maxInputPoints > 0) reserve = Mathf.Min(reserve, maxInputPoints);
        var points = new List<PointColor>(Mathf.Max(1, reserve));
        using (var ms = new MemoryStream(data, headerBytes, data.Length - headerBytes, false))
        using (var br = new BinaryReader(ms))
        {
            bool hasRgbProps = idx.r >= 0 && idx.g >= 0 && idx.b >= 0;
            bool hasFdcProps = idx.f0 >= 0 && idx.f1 >= 0 && idx.f2 >= 0;
            for (int i = 0; i < vertexCount; i++)
            {
                double x = 0, y = 0, z = 0, r = 1, g = 1, b = 1, f0 = 0, f1 = 0, f2 = 0;

                for (int p = 0; p < props.Count; p++)
                {
                    double v = ReadScalarAsDouble(br, props[p].type, littleEndian);
                    if (p == idx.x) x = v;
                    else if (p == idx.y) y = v;
                    else if (p == idx.z) z = v;
                    else if (p == idx.r) r = v;
                    else if (p == idx.g) g = v;
                    else if (p == idx.b) b = v;
                    else if (p == idx.f0) f0 = v;
                    else if (p == idx.f1) f1 = v;
                    else if (p == idx.f2) f2 = v;
                }

                if (!ShouldKeepPoint(i, points.Count))
                    continue;

                PointColor point;
                if (TryExtractPoint(x, y, z, hasRgbProps, r, g, b, hasFdcProps, f0, f1, f2, out point))
                    points.Add(point);
            }
        }
        return points;
    }

    bool TryExtractPoint(string[] values, PropertyIndices idx, out PointColor point)
    {
        point = default(PointColor);
        if (idx.x < 0 || idx.y < 0 || idx.z < 0) return false;

        if (!TryParseDouble(values[idx.x], out double x) || !TryParseDouble(values[idx.y], out double y) || !TryParseDouble(values[idx.z], out double z))
            return false;

        bool hasRGB = false;
        double r = 1, g = 1, b = 1;
        if (idx.r >= 0 && idx.g >= 0 && idx.b >= 0 &&
            TryParseDouble(values[idx.r], out r) &&
            TryParseDouble(values[idx.g], out g) &&
            TryParseDouble(values[idx.b], out b))
        {
            hasRGB = true;
        }

        bool hasFdc = false;
        double f0 = 0, f1 = 0, f2 = 0;
        if (idx.f0 >= 0 && idx.f1 >= 0 && idx.f2 >= 0 &&
            TryParseDouble(values[idx.f0], out f0) &&
            TryParseDouble(values[idx.f1], out f1) &&
            TryParseDouble(values[idx.f2], out f2))
        {
            hasFdc = true;
        }

        return TryExtractPoint(x, y, z, hasRGB, r, g, b, hasFdc, f0, f1, f2, out point);
    }

    bool TryExtractPoint(double x, double y, double z, bool hasRGB, double r, double g, double b, bool hasFdc, double f0, double f1, double f2, out PointColor point)
    {
        point = default(PointColor);
        if (!IsFinite(x) || !IsFinite(y) || !IsFinite(z)) return false;
        if (flipZ) z = -z;

        Color c = Color.white;
        if (hasRGB)
        {
            if (r > 1.0 || g > 1.0 || b > 1.0)
                c = new Color((float)(r / 255.0), (float)(g / 255.0), (float)(b / 255.0), 1f);
            else
                c = new Color((float)r, (float)g, (float)b, 1f);
        }
        else if (hasFdc)
        {
            c = new Color(
                Mathf.Clamp01((float)(0.5 + SH_C0 * f0)),
                Mathf.Clamp01((float)(0.5 + SH_C0 * f1)),
                Mathf.Clamp01((float)(0.5 + SH_C0 * f2)),
                1f
            );
        }

        point = new PointColor { pos = new Vector3((float)x, (float)y, (float)z), color = c };
        return true;
    }

    static int FindProp(List<PlyProperty> props, string name)
    {
        for (int i = 0; i < props.Count; i++)
            if (props[i].name == name) return i;
        return -1;
    }

    static bool TryParseDouble(string s, out double v) =>
        double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out v);

    static bool IsFinite(double v) => !(double.IsNaN(v) || double.IsInfinity(v));

    static string[] SplitWS(string s) =>
        s.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);

    static double ReadScalarAsDouble(BinaryReader br, string type, bool littleEndian)
    {
        switch (type)
        {
            case "char":
            case "int8":
                return br.ReadSByte();
            case "uchar":
            case "uint8":
                return br.ReadByte();
            case "short":
            case "int16":
                return ReadInt16(br, littleEndian);
            case "ushort":
            case "uint16":
                return ReadUInt16(br, littleEndian);
            case "int":
            case "int32":
                return ReadInt32(br, littleEndian);
            case "uint":
            case "uint32":
                return ReadUInt32(br, littleEndian);
            case "float":
            case "float32":
                return ReadFloat32(br, littleEndian);
            case "double":
            case "float64":
                return ReadFloat64(br, littleEndian);
            default:
                throw new InvalidDataException($"Unsupported PLY property type: {type}");
        }
    }

    static short ReadInt16(BinaryReader br, bool littleEndian)
    {
        byte[] b = br.ReadBytes(2);
        if (b.Length != 2) throw new EndOfStreamException();
        if (BitConverter.IsLittleEndian != littleEndian) Array.Reverse(b);
        return BitConverter.ToInt16(b, 0);
    }

    static ushort ReadUInt16(BinaryReader br, bool littleEndian)
    {
        byte[] b = br.ReadBytes(2);
        if (b.Length != 2) throw new EndOfStreamException();
        if (BitConverter.IsLittleEndian != littleEndian) Array.Reverse(b);
        return BitConverter.ToUInt16(b, 0);
    }

    static int ReadInt32(BinaryReader br, bool littleEndian)
    {
        byte[] b = br.ReadBytes(4);
        if (b.Length != 4) throw new EndOfStreamException();
        if (BitConverter.IsLittleEndian != littleEndian) Array.Reverse(b);
        return BitConverter.ToInt32(b, 0);
    }

    static uint ReadUInt32(BinaryReader br, bool littleEndian)
    {
        byte[] b = br.ReadBytes(4);
        if (b.Length != 4) throw new EndOfStreamException();
        if (BitConverter.IsLittleEndian != littleEndian) Array.Reverse(b);
        return BitConverter.ToUInt32(b, 0);
    }

    static float ReadFloat32(BinaryReader br, bool littleEndian)
    {
        byte[] b = br.ReadBytes(4);
        if (b.Length != 4) throw new EndOfStreamException();
        if (BitConverter.IsLittleEndian != littleEndian) Array.Reverse(b);
        return BitConverter.ToSingle(b, 0);
    }

    static double ReadFloat64(BinaryReader br, bool littleEndian)
    {
        byte[] b = br.ReadBytes(8);
        if (b.Length != 8) throw new EndOfStreamException();
        if (BitConverter.IsLittleEndian != littleEndian) Array.Reverse(b);
        return BitConverter.ToDouble(b, 0);
    }

    void BuildVoxelMesh(List<PointColor> points)
    {
        var occ = new HashSet<Vector3Int>();
        var colorAcc = new Dictionary<Vector3Int, VoxelColorAcc>();
        float inv = 1f / voxelSize;

        foreach (var p in points)
        {
            var k = new Vector3Int(
                Mathf.RoundToInt(p.pos.x * inv),
                Mathf.RoundToInt(p.pos.y * inv),
                Mathf.RoundToInt(p.pos.z * inv));

            bool existed = occ.Contains(k);
            if (!existed && maxVoxels > 0 && occ.Count >= maxVoxels)
                continue;
            if (!existed) occ.Add(k);
            VoxelColorAcc acc;
            if (!colorAcc.TryGetValue(k, out acc)) acc = new VoxelColorAcc();
            acc.Add(p.color);
            colorAcc[k] = acc;
        }

        for (int step = 0; step < dilationSteps; step++)
        {
            var toAdd = new HashSet<Vector3Int>();
            foreach (var v in occ)
                foreach (var d in kNeigh6)
                    if (!occ.Contains(v + d)) toAdd.Add(v + d);

            foreach (var v in toAdd)
            {
                occ.Add(v);
                Color c = EstimateNeighborColor(v, occ, colorAcc);
                colorAcc[v] = new VoxelColorAcc { count = 1, colorSum = new Vector3(c.r, c.g, c.b) };
            }
        }

        var verts = new List<Vector3>(occ.Count * 24);
        var tris = new List<int>(occ.Count * 36);
        var cols = new List<Color>(occ.Count * 24);
        float h = voxelSize * 0.5f;
        int triBase = 0;

        foreach (var v in occ)
        {
            Vector3 center = new Vector3(v.x * voxelSize, v.y * voxelSize, v.z * voxelSize);
            Color c = colorAcc.TryGetValue(v, out var acc) ? acc.Avg() : Color.white;

            for (int f = 0; f < 6; f++)
            {
                if (occ.Contains(v + kNeigh6[f])) continue;
                Vector3[] q = FaceQuad(center, h, f);
                verts.AddRange(q);
                cols.Add(c); cols.Add(c); cols.Add(c); cols.Add(c);
                tris.Add(triBase + 0); tris.Add(triBase + 1); tris.Add(triBase + 2);
                tris.Add(triBase + 0); tris.Add(triBase + 2); tris.Add(triBase + 3);
                triBase += 4;
            }
        }

        Mesh mesh = new Mesh { indexFormat = IndexFormat.UInt32 };
        mesh.SetVertices(verts);
        mesh.SetTriangles(tris, 0);
        mesh.SetColors(cols);
        if (recalculateNormals)
            mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        var mf = GetComponent<MeshFilter>();
        mf.sharedMesh = mesh;
        var mr = GetComponent<MeshRenderer>();
        ApplyMaterial(mr);

        if (buildCollider)
        {
            var mc = GetComponent<MeshCollider>();
            if (mc == null) mc = gameObject.AddComponent<MeshCollider>();
            mc.sharedMesh = null;
            mc.sharedMesh = mesh;
        }

        Debug.Log($"Trace3DPlyVoxelMeshRenderer: points={points.Count}, voxels={occ.Count}, triangles={tris.Count / 3}");
    }

    void ApplyMaterial(MeshRenderer mr)
    {
        if (renderMaterial != null)
        {
            mr.sharedMaterial = renderMaterial;
            return;
        }

        if (mr.sharedMaterial != null) return;

        string[] candidates =
        {
            "Universal Render Pipeline/Particles/Unlit",
            "Particles/Standard Unlit",
            "Sprites/Default",
            "Legacy Shaders/Particles/Alpha Blended",
        };

        Shader shader = null;
        for (int i = 0; i < candidates.Length; i++)
        {
            shader = Shader.Find(candidates[i]);
            if (shader != null) break;
        }
        if (shader == null) shader = Shader.Find("Standard");

        var mat = new Material(shader) { enableInstancing = true };
        mat.color = Color.white;
        mr.sharedMaterial = mat;
    }

    static Color EstimateNeighborColor(Vector3Int v, HashSet<Vector3Int> occ, Dictionary<Vector3Int, VoxelColorAcc> colorAcc)
    {
        Vector3 sum = Vector3.zero;
        int cnt = 0;
        foreach (var d in kNeigh6)
        {
            var n = v + d;
            if (occ.Contains(n) && colorAcc.TryGetValue(n, out var acc))
            {
                Color c = acc.Avg();
                sum += new Vector3(c.r, c.g, c.b);
                cnt++;
            }
        }
        if (cnt == 0) return Color.white;
        sum /= cnt;
        return new Color(sum.x, sum.y, sum.z, 1f);
    }

    static Vector3[] FaceQuad(Vector3 c, float h, int f)
    {
        switch (f)
        {
            case 0: return new[] { c + new Vector3(h, -h, -h), c + new Vector3(h, -h, h), c + new Vector3(h, h, h), c + new Vector3(h, h, -h) }; // +X
            case 1: return new[] { c + new Vector3(-h, -h, h), c + new Vector3(-h, -h, -h), c + new Vector3(-h, h, -h), c + new Vector3(-h, h, h) }; // -X
            case 2: return new[] { c + new Vector3(-h, h, -h), c + new Vector3(h, h, -h), c + new Vector3(h, h, h), c + new Vector3(-h, h, h) }; // +Y
            case 3: return new[] { c + new Vector3(-h, -h, h), c + new Vector3(h, -h, h), c + new Vector3(h, -h, -h), c + new Vector3(-h, -h, -h) }; // -Y
            case 4: return new[] { c + new Vector3(h, -h, h), c + new Vector3(-h, -h, h), c + new Vector3(-h, h, h), c + new Vector3(h, h, h) }; // +Z
            default: return new[] { c + new Vector3(-h, -h, -h), c + new Vector3(h, -h, -h), c + new Vector3(h, h, -h), c + new Vector3(-h, h, -h) }; // -Z
        }
    }
}
