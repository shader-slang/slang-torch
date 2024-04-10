#include "BRDF.slang"

[Differentiable]
float bilin_weight(float t)
{
    if (abs(t) > 1.0f)
        return 0.0f;

    return 1.0f - abs(t);
}

[Differentiable]
float getInputElement(DiffTensorView<float> input,
                      no_diff int3 loc)
{
    loc = min(loc, int3(input.size(0) - 1, input.size(1) - 1, input.size(2) - 1));
    loc = max(loc, int3(0, 0, 0));
    return input.load(loc);
}

[Differentiable]
float sampleBilinear(DiffTensorView<float> input,
                     no_diff float2 uv,
                     no_diff int z_slice)
{
    float2 accum = 0.0f;
    float2 pixel_coord = uv * float2(input.size(0), input.size(1)) - 0.5;
    int2 pixel_coord_int = int2(floor(pixel_coord));
    float2 frac_offset = no_diff float2(pixel_coord_int) - pixel_coord;
    for (int dy = 0; dy <= 1; dy++)
    {
        for (int dx = 0; dx <= 1; dx++)
        {
            int2 finalCoordFloat = pixel_coord_int + int2(dx, dy);
            float2 relativeDiff = frac_offset + float2(dx, dy);
            float w = bilin_weight(relativeDiff.x) * bilin_weight(relativeDiff.y);

            accum += float2(getInputElement(input, int3(finalCoordFloat, z_slice)), 1.0f) * w;
        }
    }
    return accum.x;
}

struct Params
{
    float3 L;
    float3 V;
};

struct InputParams
{
    float L_x, L_y, L_z;
    float V_x, V_y, V_z;
};

Params PackInputParams(InputParams input_params)
{
    Params params;
    params.L = normalize(float3(input_params.L_x, input_params.L_y, input_params.L_z));
    params.V = normalize(float3(input_params.V_x, input_params.V_y, input_params.V_z));
    return params;
}

[Differentiable]
float3 computeLighting(DiffTensorView<float> input,
                       float2 uv,
                       no_diff Params params)
{
    float3 baseColor, N;
    float roughness;
    baseColor = float3(sampleBilinear(input, uv, 0),
                       sampleBilinear(input, uv, 1),
                       sampleBilinear(input, uv, 2));

    N = float3(sampleBilinear(input, uv, 3),
               sampleBilinear(input, uv, 4),
               sampleBilinear(input, uv, 5));

    roughness = sampleBilinear(input, uv, 6);
    baseColor = baseColor * baseColor;
    N = N - float3(0.5f, 0.5f, 0.0f);
    N = normalize(N);

    float3 lighting = BRDF(params.L, params.V, N, baseColor, roughness);

    return lighting;
}

[AutoPyBindCUDA]
[CudaKernel]
void brdf(TensorView<float> input, TensorView<float> output, InputParams input_params)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x >= output.size(0) ||
        globalIdx.y >= output.size(1))
        return;

    float2 uv = (globalIdx.xy + 0.5) / float2(output.size(0), output.size(1));
    DiffTensorView view;
    view.primal = input;

    float3 lighting = computeLighting(view, uv, PackInputParams(input_params));

    output[int3(globalIdx.xy, 0)] = sqrt(lighting.r);
    output[int3(globalIdx.xy, 1)] = sqrt(lighting.g);
    output[int3(globalIdx.xy, 2)] = sqrt(lighting.b);
}

[Differentiable]
float loss(DiffTensorView<float> input,
           no_diff float2 uv,
           no_diff float3 reference,
           no_diff Params params)
{
    float3 diff = sqrt(max(computeLighting(input, uv, params), 0.0f)) - reference;
    return dot(diff, diff);
}

[AutoPyBindCUDA]
[CudaKernel]
[Differentiable]
void brdf_loss(DiffTensorView<float> input, DiffTensorView<float> output, TensorView<float> reference, InputParams input_params)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x >= output.size(0) ||
        globalIdx.y >= output.size(1))
        return;

    float2 uv = (globalIdx.xy + 0.5) / float2(output.size(0), output.size(1));
    float3 referenceVal = float3(reference[int3(globalIdx.xy, 0)], reference[int3(globalIdx.xy, 1)], reference[int3(globalIdx.xy, 2)]);
    float loss = loss(input, uv, referenceVal, PackInputParams(input_params));

    output.store(globalIdx, loss);
}
