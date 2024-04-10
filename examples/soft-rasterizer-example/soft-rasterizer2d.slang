
[PreferRecompute]
int _sign(float x)
{
    return sign(x);
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
    [Differentiable]
    float2 screenToWorld(float2 uv)
    {
        float2 xy = uv / frameDim;
        float2 ndc = xy * 2.0f - 1.0f;
        return ndc * scale + o;
    }
};

[Differentiable]
[PreferRecompute]
float distanceToEdge(float2 u, float2 v, float2 pt)
{
    float2 e = v - u;
    float2 n = float2(-e.y, e.x);
    float2 d = pt - u;
    float n_dot_d = dot(n, d);

    // Compute the signed distance from a point to an edge
    if (dot(pt - u, v - u) < 0.f)
        return distanceToVertex(pt, u) * (_sign(n_dot_d)); // u is the closest vertex
    else if (dot(pt - v, u - v) < 0.f)
        return distanceToVertex(pt, v) * (_sign(n_dot_d)); // v is the closest vertex
    else
    {
        return n_dot_d / length(n); // closest edge
    }
}

[Differentiable]
[PreferRecompute]
float distanceToVertex(float2 xy2, float2 v)
{
    // Compute the distance from a point to a vertex
    return length(xy2 - v);
}

[Differentiable]
[PreferRecompute]
float distanceToTriangle(float2 xy, float2 v1, float2 v2, float2 v3)
{
    // Minimum distance to the edge of the triangle
    float d1 = distanceToEdge(v2, v1, xy);
    float d2 = distanceToEdge(v3, v2, xy);
    float d3 = distanceToEdge(v1, v3, xy);

    int sd1 = _sign(d1);
    int sd2 = _sign(d2);
    int sd3 = _sign(d3);

    if (sd1 > 0.f && sd2 > 0.f)
        return distanceToVertex(xy, v2); // v2 is the closest vertex
    else if (sd1 > 0.f && sd3 > 0.f)
        return distanceToVertex(xy, v1); // v1 is the closest vertex
    else if (sd2 > 0.f && sd3 > 0.f)
        return distanceToVertex(xy, v3); // v3 is the closest vertex
    else 
        return max(max(d1, d2), d3);

}

[Differentiable]
[PreferRecompute]
float sigmoid(float x, float sigma)
{
    // Sigmoid function
    return 1.0 / (1.0 + exp(-x / sigma));
}

[Differentiable]
[PreferRecompute]
float softTriangle(float2 xy, float2 v1, float2 v2, float2 v3, float sigma)
{
    float d = distanceToTriangle(xy, v1, v2, v3);
    return sigmoid(-d, sigma);
}

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void rasterize(
    Camera camera,
    DiffTensorView vertices,
    DiffTensorView color,
    DiffTensorView output)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x >= output.size(0) || globalIdx.y >= output.size(1))
        return;

    // Load vertices of our triangle.
    // Assume our input tensor is of the form (3, 2) where 3 is the number of vertices
    // and 2 is the number of coordinates per vertex.
    //
    float2 v1 = float2(vertices[uint2(0, 0)], vertices[uint2(0, 1)]);
    float2 v2 = float2(vertices[uint2(1, 0)], vertices[uint2(1, 1)]);
    float2 v3 = float2(vertices[uint2(2, 0)], vertices[uint2(2, 1)]);
    float3 c = float3(color[0], color[1], color[2]);
    float3 background_color = float3(1.f);

    // Sigma for the sigmoid function
    float sigma = 0.02;

    // Compute result for the current pixel.
    float2 screen_sample = globalIdx.xy + 0.5;
    float2 world_sample = camera.screenToWorld(screen_sample);

    float hit = softTriangle(world_sample, v1, v2, v3, sigma);

    float3 result = hit * c + (1 - hit) * background_color;
    
    // Write-back using the 'storeOnce' method that has a more efficient 
    // derivative implementation if each index is written to only once.
    //
    output.storeOnce(uint3(globalIdx.xy, 0), result.x);
    output.storeOnce(uint3(globalIdx.xy, 1), result.y);
    output.storeOnce(uint3(globalIdx.xy, 2), result.z);

}