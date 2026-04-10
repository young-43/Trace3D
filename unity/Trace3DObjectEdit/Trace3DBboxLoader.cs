using System;
using System.Collections.Generic;
using UnityEngine;

public class Trace3DBboxLoader : MonoBehaviour
{
    [Header("Input")]
    public TextAsset bboxJson;

    [Header("Spawn")]
    public Material boxMaterial;
    public Transform root;
    public bool useOrientedBox = true;

    [Serializable]
    public class Vec3Data { public float[] center; public float[] size; }

    [Serializable]
    public class ObbData
    {
        public float[] center;
        public float[] size;
        public float[] quaternion_xyzw;
    }

    [Serializable]
    public class UnityBoxData
    {
        public Vec3Data aabb;
        public ObbData obb;
    }

    [Serializable]
    public class ObjectData
    {
        public string object_name;
        public string mask_file;
        public int gaussian_count;
        public Vec3Data aabb;
        public ObbData obb;
        public UnityBoxData unity;
    }

    [Serializable]
    public class BboxFileData
    {
        public int version;
        public ObjectData[] objects;
    }

    private readonly List<GameObject> spawned = new List<GameObject>();

    [ContextMenu("Load Boxes")]
    public void LoadBoxes()
    {
        ClearBoxes();
        if (bboxJson == null)
        {
            Debug.LogError("Trace3DBboxLoader: bboxJson is null");
            return;
        }

        var data = JsonUtility.FromJson<BboxFileData>(bboxJson.text);
        if (data == null || data.objects == null)
        {
            Debug.LogError("Trace3DBboxLoader: failed to parse bbox json");
            return;
        }

        Transform parent = root != null ? root : transform;

        foreach (var obj in data.objects)
        {
            if (obj == null || obj.unity == null) continue;
            var boxData = useOrientedBox ? obj.unity.obb : ToObbLike(obj.unity.aabb);
            if (boxData == null) continue;

            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = $"bbox_{obj.object_name}";
            go.transform.SetParent(parent, worldPositionStays: false);
            go.transform.position = ToVector3(boxData.center);
            go.transform.localScale = ToVector3(boxData.size);
            go.transform.rotation = ToQuaternion(boxData.quaternion_xyzw);

            var renderer = go.GetComponent<MeshRenderer>();
            if (renderer != null && boxMaterial != null)
            {
                renderer.sharedMaterial = boxMaterial;
            }

            var drag = go.AddComponent<Trace3DBoxDrag>();
            drag.Label = obj.object_name;

            spawned.Add(go);
        }

        Debug.Log($"Trace3DBboxLoader: loaded {spawned.Count} boxes");
    }

    [ContextMenu("Clear Boxes")]
    public void ClearBoxes()
    {
        for (int i = 0; i < spawned.Count; ++i)
        {
            if (spawned[i] != null) DestroyImmediate(spawned[i]);
        }
        spawned.Clear();
    }

    private static Vector3 ToVector3(float[] v)
    {
        if (v == null || v.Length < 3) return Vector3.zero;
        return new Vector3(v[0], v[1], v[2]);
    }

    private static Quaternion ToQuaternion(float[] q)
    {
        if (q == null || q.Length < 4) return Quaternion.identity;
        return new Quaternion(q[0], q[1], q[2], q[3]);
    }

    private static ObbData ToObbLike(Vec3Data aabb)
    {
        if (aabb == null) return null;
        return new ObbData
        {
            center = aabb.center,
            size = aabb.size,
            quaternion_xyzw = new[] { 0f, 0f, 0f, 1f }
        };
    }
}
