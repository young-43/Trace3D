using UnityEngine;

[RequireComponent(typeof(Collider))]
public class Trace3DBoxDrag : MonoBehaviour
{
    public string Label = "object";

    private Camera mainCamera;
    private bool dragging;
    private float depth;
    private Vector3 offset;

    private void Awake()
    {
        mainCamera = Camera.main;
    }

    private void OnMouseDown()
    {
        if (mainCamera == null) mainCamera = Camera.main;
        if (mainCamera == null) return;

        var screen = mainCamera.WorldToScreenPoint(transform.position);
        depth = screen.z;
        offset = transform.position - ScreenToWorldPoint(Input.mousePosition, depth);
        dragging = true;
    }

    private void OnMouseDrag()
    {
        if (!dragging || mainCamera == null) return;
        transform.position = ScreenToWorldPoint(Input.mousePosition, depth) + offset;
    }

    private void OnMouseUp()
    {
        dragging = false;
        var p = transform.position;
        Debug.Log($"Trace3DBoxDrag [{Label}] moved to ({p.x:F4}, {p.y:F4}, {p.z:F4})");
    }

    private Vector3 ScreenToWorldPoint(Vector3 mousePos, float zDepth)
    {
        return mainCamera.ScreenToWorldPoint(new Vector3(mousePos.x, mousePos.y, zDepth));
    }
}
