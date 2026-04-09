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

    [Header("Voxel Mesh")]
    [Min(0.0001f)] public float voxelSize = 0.02f;
    [Range(0, 2)] public int dilationSteps = 1;
    public bool flipZ = true;

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

        switch (format)
        {
            case PlyFormat.Ascii:
                return ReadAsciiVertices(data, headerBytes, vertexCount, properties);
            case PlyFormat.BinaryLittleEndian:
                return ReadBinaryVertices(data, headerBytes, vertexCount, properties, true);
            case PlyFormat.BinaryBigEndian:
                return ReadBinaryVertices(data, headerBytes, vertexCount, properties, false);
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
                inVertexElement = t.Length >= 3 && t[1] == "vertex";
                if (inVertexElement && !int.TryParse(t[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out vertexCount))
                    throw new InvalidDataException($"Invalid vertex element line: {line}");
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
                    throw new InvalidDataException("Vertex list properties are not supported.");
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

    List<PointColor> ReadAsciiVertices(byte[] data, int headerBytes, int vertexCount, List<PlyProperty> props)
    {
        string body = Encoding.UTF8.GetString(data, headerBytes, data.Length - headerBytes);
        var points = new List<PointColor>(vertexCount);
        using (var sr = new StringReader(body))
        {
            string line;
            int readCount = 0;
            while (readCount < vertexCount && (line = sr.ReadLine()) != null)
            {
                line = line.Trim();
                if (line.Length == 0) continue;
                string[] t = SplitWS(line);
                if (t.Length < props.Count) continue;
                PointColor p;
                if (TryExtractPoint(t, props, out p)) points.Add(p);
                readCount++;
            }
        }
        return points;
    }

    List<PointColor> ReadBinaryVertices(byte[] data, int headerBytes, int vertexCount, List<PlyProperty> props, bool littleEndian)
    {
        var points = new List<PointColor>(vertexCount);
        using (var ms = new MemoryStream(data, headerBytes, data.Length - headerBytes, false))
        using (var br = new BinaryReader(ms))
        {
            for (int i = 0; i < vertexCount; i++)
            {
                var values = new string[props.Count];
                for (int p = 0; p < props.Count; p++)
                {
                    double v = ReadScalarAsDouble(br, props[p].type, littleEndian);
                    values[p] = v.ToString("R", CultureInfo.InvariantCulture);
                }

                PointColor point;
                if (TryExtractPoint(values, props, out point))
                    points.Add(point);
            }
        }
        return points;
    }

    bool TryExtractPoint(string[] values, List<PlyProperty> props, out PointColor point)
    {
        point = default;
        int ix = FindProp(props, "x"), iy = FindProp(props, "y"), iz = FindProp(props, "z");
        if (ix < 0 || iy < 0 || iz < 0) return false;

        if (!TryParseDouble(values[ix], out double x) || !TryParseDouble(values[iy], out double y) || !TryParseDouble(values[iz], out double z))
            return false;

        if (!IsFinite(x) || !IsFinite(y) || !IsFinite(z)) return false;
        if (flipZ) z = -z;

        Color c = Color.white;
        int ir = FindProp(props, "red"), ig = FindProp(props, "green"), ib = FindProp(props, "blue");
        if (ir >= 0 && ig >= 0 && ib >= 0 &&
            TryParseDouble(values[ir], out double r) &&
            TryParseDouble(values[ig], out double g) &&
            TryParseDouble(values[ib], out double b))
        {
            if (r > 1.0 || g > 1.0 || b > 1.0) c = new Color((float)(r / 255.0), (float)(g / 255.0), (float)(b / 255.0), 1f);
            else c = new Color((float)r, (float)g, (float)b, 1f);
        }
        else
        {
            int if0 = FindProp(props, "f_dc_0"), if1 = FindProp(props, "f_dc_1"), if2 = FindProp(props, "f_dc_2");
            if (if0 >= 0 && if1 >= 0 && if2 >= 0 &&
                TryParseDouble(values[if0], out double f0) &&
                TryParseDouble(values[if1], out double f1) &&
                TryParseDouble(values[if2], out double f2))
            {
                c = new Color(
                    Mathf.Clamp01((float)(0.5 + SH_C0 * f0)),
                    Mathf.Clamp01((float)(0.5 + SH_C0 * f1)),
                    Mathf.Clamp01((float)(0.5 + SH_C0 * f2)),
                    1f
                );
            }
        }

        point = new PointColor
        {
            pos = new Vector3((float)x, (float)y, (float)z),
            color = c
        };
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

            occ.Add(k);
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
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        var mf = GetComponent<MeshFilter>();
        mf.sharedMesh = mesh;
        var mr = GetComponent<MeshRenderer>();
        if (mr.sharedMaterial == null)
        {
            var mat = new Material(Shader.Find("Standard")) { enableInstancing = true };
            mr.sharedMaterial = mat;
        }

        if (buildCollider)
        {
            var mc = GetComponent<MeshCollider>();
            if (mc == null) mc = gameObject.AddComponent<MeshCollider>();
            mc.sharedMesh = null;
            mc.sharedMesh = mesh;
        }

        Debug.Log($"Trace3DPlyVoxelMeshRenderer: points={points.Count}, voxels={occ.Count}, triangles={tris.Count / 3}");
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
