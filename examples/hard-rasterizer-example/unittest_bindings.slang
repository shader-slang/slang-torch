import hard_rasterizer2d;

[AutoPyBindCUDA]
[CudaKernel]
void aabb_intersection_direct_call(
    TensorView<float2> aabb,
    TensorView<float2> segment,
    TensorView<bool> outValid,
    TensorView<float2> outNear,
    TensorView<float2> outFar)
{
    // Call the function from hard_rasterizer2d.
    AABB _aabb = { aabb[0], aabb[1] };
    float2 _a = segment[0];
    float2 _b = segment[1];
    Maybe<Intersection> intersection = intersectSegmentAABB(_aabb, _a, _b);

    // Write the results.
    outValid[0] = intersection.is_valid;
    outNear[0] = intersection.value.isect_near;
    outFar[0] = intersection.value.isect_far;
}

[AutoPyBindCUDA]
[CudaKernel]
void triangle_sample_from_edge_direct_call(
    TensorView<float2> vertices,
    TensorView<float2> aabb,
    TensorView<float> sample1D,
    TensorView<bool> outValid,
    TensorView<float2> outPt,
    TensorView<float2> outNormal
    )
{
    Triangle triangle = { vertices[0], vertices[1], vertices[1], float3(1.f), 0.02 };

    Maybe<EdgeSample> es_maybe = triangle.sampleFromEdge(
        vertices[0], vertices[1], sample1D[0], {aabb[0], aabb[1]});

    outValid[0] = es_maybe.is_valid;
    outPt[0] = es_maybe.value.pt;
    outNormal[0] = es_maybe.value.n;
}

[AutoPyBindCUDA]
[CudaKernel]
void triangle_sample_from_boundary_direct_call(
    TensorView<float2> vertices,
    TensorView<float2> aabb,
    TensorView<float> sample1D,
    TensorView<bool> outValid,
    TensorView<float2> outPt,
    TensorView<float2> outNormal
    )
{
    // Make triangle.
    Triangle triangle = { vertices[0], vertices[1], vertices[2], float3(1.f), 0.02 };

    Maybe<EdgeSample> es_maybe = triangle.sampleFromBoundary(sample1D[0], { aabb[0], aabb[1] });

    // Write the results.
    outValid[0] = es_maybe.is_valid;
    outPt[0] = es_maybe.value.pt;
    outNormal[0] = es_maybe.value.n;
}

[AutoPyBindCUDA]
[CudaKernel]
[Differentiable]
void render_pixel_direct_call(
    DiffTensorView vertices,
    DiffTensorView pixel,
    TensorView<int> rngState,
    DiffTensorView output
    )
{
    Triangle triangle = { { vertices[0, 0], vertices[0, 1] },
                          { vertices[1, 0], vertices[1, 1] }, 
                          { vertices[2, 0], vertices[2, 1] },
                          float3(1.f, 0.f, 0.f), 0.02 };
    Camera camera = { float2(0.0, 0.0), float2(1.0, 1.0), float2(256, 256) };
    PRNG prng = PRNG(reinterpret<uint, int>(rngState[0]));

    float3 color = render_pixel( { pixel[0], pixel[1] }, camera, triangle, prng);
    output[0] = color[0];
    output[1] = color[1];
    output[2] = color[2];
}