
// xy: 2D test position
// v1: vertex position 1
// v2: vertex position 2
// v3: vertex position 3
//
bool triangle(float2 xy, float2 v1, float2 v2, float2 v3)
{
    // 2D triangle test (return 1.0 if the triangle contains the point (x,y), 0.0 otherwise).
    float2 e0 = v2 - v1;
    float2 e1 = v3 - v2;
    float2 e2 = v1 - v3;

    float2 k0 = float2(-e0.y, e0.x);
    float2 k1 = float2(-e1.y, e1.x);
    float2 k2 = float2(-e2.y, e2.x);

    float d0 = dot(k0, xy - v1);
    float d1 = dot(k1, xy - v2);
    float d2 = dot(k2, xy - v3);

    // Check the three half-plane values.
    if(d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0)
        return true;
    else
        return false;
}

struct Camera
{
    // World origin
    float2 o;

    // World scale
    float2 scale;

    // Frame dimensions (i.e. image resolution)
    float2 frameDim;

    // Convert from
    // screen coordinates [(0, 0), (W, H)] to
    // world coordinates [(o.x - scale.x, o.y - scale.y), (o.x + scale.x, o.y + scale.y)]
    //
    float2 screenToWorld(float2 uv)
    {
        float2 xy = uv / frameDim;
        float2 ndc = xy * 2.0f - 1.0f;
        return ndc * scale + o;
    }
};

[AutoPyBindCUDA]
[CUDAKernel]
void rasterize(
    Camera camera,
    TensorView<float> vertices,
    TensorView<float> color,
    TensorView<float3> output)
{
    uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (dispatch_id.x > output.size(0) || dispatch_id.y > output.size(1))
        return;

    float2 v1 = float2(vertices[0, 0], vertices[0, 1]);
    float2 v2 = float2(vertices[1, 0], vertices[1, 1]);
    float2 v3 = float2(vertices[2, 0], vertices[2, 1]);
    float3 c = float3(color[0], color[1], color[2]);
    float3 background_color = float3(1.0, 1.0, 1.0);

    float2 pixel_coord = dispatch_id.xy;
    float2 screen_sample = pixel_coord + 0.5;

    // Convert screen space pixel coordinates to world space.
    float2 world_sample = camera.screenToWorld(screen_sample);

    bool hit = triangle(world_sample, v1, v2, v3);
    float3 result = hit ? c : background_color;
    output[dispatch_id.xy] = result;
}