
struct PRNG
{
    __init(uint seed)
    {
        this.state = seed;
    }

    [mutating] uint next()
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    [mutating] float nextFloat1D()
    {
        return float(next()) / float(4294967295.0);
    }

    uint state;
};

[PreferRecompute]
int _sign(float x)
{
    return sign(x);
}

struct Intersection : IDifferentiable
{
    float2 isect_near;
    float2 isect_far;
};

struct Camera
{
    float2 o;
    float2 scale;
    float2 frameDim;

    [Differentiable]
    [PreferRecompute]
    float2 screenToWorld(float2 uv)
    {
        // Convert screen coordinates to world coordinates
        float2 xy = uv / frameDim;
        float2 ndc = xy * 2.0f - 1.0f;
        return ndc * scale + o;
    }

    [Differentiable]
    [PreferRecompute]
    float2 worldToScreen(float2 xy)
    {
        // Convert world coordinates to screen coordinates
        float2 ndc = (xy - o) / scale;
        float2 uv = (ndc + 1.0f) / 2.0f;
        return uv * frameDim;
    }

    [Differentiable]
    [PreferRecompute]
    AABB screenToWorld(AABB aabb)
    {
        return { screenToWorld(aabb.low), screenToWorld(aabb.high) };
    }

    [Differentiable]
    [PreferRecompute]
    AABB worldToScreen(AABB aabb)
    {
        return { worldToScreen(aabb.low), worldToScreen(aabb.high) };
    }
};

struct AABB : IDifferentiable
{
    float2 low;
    float2 high;
};

struct EdgeSample : IDifferentiable
{
    float2 pt;
    float2 n;
    float weight;
};

struct Maybe<T : IDifferentiable> : IDifferentiable
{
    bool is_valid;
    T value;

    [Differentiable]
    static Maybe<T> none()
    {
        Maybe<T> result;
        result.is_valid = false;
        return result;
    }

    [Differentiable]
    static Maybe<T> some(T value)
    {
        return {true, value};
    }
};

[PreferRecompute]
Maybe<Intersection> intersectSegmentAABB(AABB aabb, float2 u, float2 v)
{
    float2 box_pos = (aabb.low + aabb.high) * 0.5;
    float2 box_half = (aabb.high - aabb.low) * 0.5;

    float2 delta = v - u;
    float scaleX = 1.0 / delta.x;
    float scaleY = 1.0 / delta.y;
    float signX = _sign(scaleX);
    float signY = _sign(scaleY);
    float nearTimeX = (box_pos.x - signX * box_half.x - u.x) * scaleX;
    float nearTimeY = (box_pos.y - signY * box_half.y - u.y) * scaleY;
    float farTimeX = (box_pos.x + signX * box_half.x - u.x) * scaleX;
    float farTimeY = (box_pos.y + signY * box_half.y - u.y) * scaleY;

    if (nearTimeX > farTimeY || nearTimeY > farTimeX)
        return Maybe<Intersection>::none();

    float nearTime = nearTimeX > nearTimeY ? nearTimeX : nearTimeY;
    float farTime = farTimeX < farTimeY ? farTimeX : farTimeY;
    
    if (nearTime >= 1 || farTime <= 0)
        return Maybe<Intersection>::none();

    if (isnan(nearTime) || isnan(farTime))
        return Maybe<Intersection>::none();

    return Maybe<Intersection>::some({u + nearTime * delta, u + farTime * delta});
}

struct Triangle : IDifferentiable
{
    float2 v1;
    float2 v2;
    float2 v3;
    float3 color;
    float sigma;

    [PreferRecompute]
    bool isInside(float2 xy)
    {
        // 2D triangle test (return 1.0 if the triangle contains the point (x,y), 0.0 otherwise)
        float2 e0 = v2 - v1;
        float2 e1 = v3 - v2;
        float2 e2 = v1 - v3;
        float2 k0 = float2(-e0.y, e0.x);
        float2 k1 = float2(-e1.y, e1.x);
        float2 k2 = float2(-e2.y, e2.x);

        float d0 = dot(k0, xy - v1);
        float d1 = dot(k1, xy - v2);
        float d2 = dot(k2, xy - v3);
        return bool(d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0);
    }

    [Differentiable]
    [PreferRecompute]
    float4 shade(float2 xy)
    {
        return float4(color, isInside(xy) ? 1.0 : 0.0);
    }

    [Differentiable]
    [PreferRecompute]
    Maybe<EdgeSample> sampleFromEdge(float2 a, float2 b, float sample1D, AABB bboxWorld)
    {
        Maybe<Intersection> isect_maybe = no_diff(intersectSegmentAABB(bboxWorld, a, b));
        if (!isect_maybe.is_valid)
            return Maybe<EdgeSample>::none();
        
        let isect = isect_maybe.value;

        // Length of the intersected segment is the wt.
        float isect_weight = length(isect.isect_far - isect.isect_near);

        // Sample from intersected segment.
        float2 segment_sample = isect.isect_near + sample1D * (isect.isect_far - isect.isect_near);

        // Relative position of the sample on the segment.
        float relpos = dot(b - a, segment_sample - a) / dot(b - a, b - a);

        // Construct sample on the edge.
        float2 edge_sample = detach(relpos) * b + (1.0 - detach(relpos)) * a;

        // Construct normal orientation for the sample.
        float2 t = a - b;
        float2 n = normalize(float2(-t.y, t.x));

        return Maybe<EdgeSample>::some({edge_sample, n, isect_weight});
    }

    [Differentiable]
    [PreferRecompute]
    Maybe<EdgeSample> sampleFromBoundary(float sample1D, AABB bboxWorld)
    {
        // Split sample1D into 6 divisions, and stretch each division to [0,1)
        // The index of the division acts as a die roll to pick between the three edges.
        // We use 6 since its a multiple of both 2 and 3, allowing us to use the same sample
        // for both the 2- and 3-edge cases.
        //
        int outerSample = (int)(floor(sample1D * 6));
        float innerSample1D = (fmod(sample1D, 1.f/6.f)) * 6.f;

        // Sample from three edges of the triangle.
        Maybe<EdgeSample> sample1 = sampleFromEdge(v1, v2, innerSample1D, bboxWorld);
        Maybe<EdgeSample> sample2 = sampleFromEdge(v2, v3, innerSample1D, bboxWorld);
        Maybe<EdgeSample> sample3 = sampleFromEdge(v3, v1, innerSample1D, bboxWorld);

        // How many edges did we hit?
        int hit_count = int(sample1.is_valid) + int(sample2.is_valid) + int(sample3.is_valid);
        
        // Pick between the three samples randomly.
        switch (hit_count)
        {
        case 0:
            return Maybe<EdgeSample>::none();
        case 1:
            return (sample1.is_valid) ? sample1 : (sample2.is_valid ? sample2 : sample3);
        case 2:
        {
            var _s1 = sample1.is_valid ? sample1 : sample2;
            var _s2 = sample1.is_valid ? (sample2.is_valid ? sample2 : sample3) : sample3;
            return (outerSample % 2 == 0) ? _s1 : _s2;
        }
        case 3:
        {
            Maybe<EdgeSample> _s[3] = { sample1, sample2, sample3 };
            return _s[(outerSample % 3)];
        }
        default:
            return Maybe<EdgeSample>::none();
        }
    }
};

[Differentiable]
[PreferRecompute]
float2 infinitesimal(float2 v)
{
    return v - detach(v);    
}

[Differentiable]
[PreferRecompute]
float3 render_pixel(no_diff float2 pixel, 
                    Camera camera,
                    Triangle triangle,
                    inout PRNG prng)
{
    // Create 4 samples around the pixel center
    float2 samples[4];
    samples[0] = pixel + float2(-1, -1) * 0.25 + 0.5; // +0.5 to center the sample
    samples[1] = pixel + float2(-1, +1) * 0.25 + 0.5;
    samples[2] = pixel + float2(+1, -1) * 0.25 + 0.5;
    samples[3] = pixel + float2(+1, +1) * 0.25 + 0.5;

    float4 result = 0.0f;

    // Compute the interior integral
    [ForceUnroll]
    for (int i = 0; i < 4; ++i)
    {
        float2 screen_sample = samples[i];
        float2 world_sample = camera.screenToWorld(screen_sample);
        result += triangle.shade(world_sample);
    }

    result = result / 4.f;

    // Compute the boundary integral.
    // Check all three edges of the triangle for intersections with the pixel.
    // If an intersection is found, compute the segment length, and take a
    // representative sample.
    //
    float sample_1d = no_diff prng.nextFloat1D();
    var bsample_maybe = triangle.sampleFromBoundary(
        sample_1d,
        camera.screenToWorld({ pixel - 0.0f, pixel + 1.0f }));
        
    if (bsample_maybe.is_valid)
    {
        EdgeSample bsample = bsample_maybe.value;
        float2 weighted_pt = camera.worldToScreen(bsample.pt) * bsample.weight;

        let eps = 0.001f;
        float4 delta_contrib = detach(
            triangle.shade(bsample.pt - eps * bsample.n) -
            triangle.shade(bsample.pt + eps * bsample.n));
        float4 b_result = dot(infinitesimal(weighted_pt), bsample.n) * delta_contrib;
        result += b_result;
    }

    // Blend with the background color.
    float3 background_color = float3(1.f);
    float3 blended_color = (1.0f - result.w) * background_color + result.w * result.xyz;

    return blended_color;
}

// ---- Boilerplate code for kernel entry point + launch parameters ----

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void rasterize(
    DiffTensorView vertices,
    DiffTensorView color,
    DiffTensorView output,
    TensorView<int> rng_state)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x > output.size(0) || globalIdx.y > output.size(1))
        return;

    PRNG prng = PRNG(reinterpret<uint, int>(rng_state[globalIdx.xy]));

    // Load vertices of our triangle.
    // Assume our input tensor is of the form (3, 2) where 3 is the number of vertices
    // and 2 is the number of coordinates per vertex.
    //
    float2 v1 = float2(vertices[uint2(0, 0)], vertices[uint2(0, 1)]);
    float2 v2 = float2(vertices[uint2(1, 0)], vertices[uint2(1, 1)]);
    float2 v3 = float2(vertices[uint2(2, 0)], vertices[uint2(2, 1)]);
    float3 c = float3(color[0], color[1], color[2]);
    
    // Setup default image parameters..
    Camera camera = { float2(0.0, 0.0), float2(1.0, 1.0), float2(output.size(0), output.size(1)) };

    Triangle triangle = { v1, v2, v3, c, 0.02 };

    // Compute value for each pixel.
    float3 color = render_pixel(globalIdx.xy, camera, triangle, prng);
    output.storeOnce(uint3(globalIdx.xy, 0), color.x);
    output.storeOnce(uint3(globalIdx.xy, 1), color.y);
    output.storeOnce(uint3(globalIdx.xy, 2), color.z);

    // Update the random number generator state.
    rng_state[globalIdx.xy] = reinterpret<int, uint>(prng.state);
}
