using UnityEngine;

[RequireComponent(typeof(Collider))]
public class Trace3DBoxDrag : MonoBehaviour
{
    public string Label = "object";

    private Camera cam;
    private bool dragging;
    private float depth;
    private Vector3 offset;

    private void Awake()
    {
        cam = Camera.main;
    }

    private void OnMouseDown()
    {
        if (cam == null) cam = Camera.main;
        if (cam == null) return;

        var screen = cam.WorldToScreenPoint(transform.position);
        depth = screen.z;
        offset = transform.position - ScreenToWorld(Input.mousePosition, depth);
        dragging = true;
    }

    private void OnMouseDrag()
    {
        if (!dragging || cam == null) return;
        transform.position = ScreenToWorld(Input.mousePosition, depth) + offset;
    }

    private void OnMouseUp()
    {
        dragging = false;
        var p = transform.position;
        Debug.Log($"Trace3DBoxDrag [{Label}] moved to ({p.x:F4}, {p.y:F4}, {p.z:F4})");
    }

    private Vector3 ScreenToWorld(Vector3 mousePos, float zDepth)
    {
        return cam.ScreenToWorldPoint(new Vector3(mousePos.x, mousePos.y, zDepth));
    }
}
