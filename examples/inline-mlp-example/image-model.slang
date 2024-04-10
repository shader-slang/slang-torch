import inline_mlp;

#define PI 3.14159265358979323846f

static const uint C = 16;

[Differentiable]
Feature<C> computeInterpolatedFeature(DiffTensorView featureGrid, uint2 frameDim)
{
    // Compute pixel block covered by each grid cell.
    uint dim0 = featureGrid.size(0) - 1;
    uint dim1 = featureGrid.size(1) - 1;

    uint2 cellSize = uint2(frameDim.x / dim0, frameDim.y / dim1);

    uint2 pixelId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).xy;

    uint2 cellId = pixelId / cellSize;

    // Compute the interpolation weights.
    float2 weights = (float2(pixelId) / cellSize) - float2(cellId);

    // Compute the interpolated feature.
    Feature<C> feature;

    [ForceUnroll]
    for (int i = 0; i < C-2; i++)
    {
        feature.vals[i] = featureGrid.load(uint3(cellId.x, cellId.y, i)) * (1 - weights.x) * (1 - weights.y) +
                          featureGrid.load(uint3(cellId.x + 1, cellId.y, i)) * weights.x * (1 - weights.y) +
                          featureGrid.load(uint3(cellId.x, cellId.y + 1, i)) * (1 - weights.x) * weights.y +
                          featureGrid.load(uint3(cellId.x + 1, cellId.y + 1, i)) * weights.x * weights.y;
    }

    feature.vals[C-2] = weights.x;
    feature.vals[C-1] = weights.y;

    return feature;
}

[Differentiable]
[CUDAKernel]
[AutoPyBindCUDA]
void renderImage(MLP<C, 3> mlp, DiffTensorView featureGrid, DiffTensorView imageOutput)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= imageOutput.size(0) || dispatchIdx.y >= imageOutput.size(1))
        return;

    // Compute the input feature for our current pixel.
    Feature<C> feature = computeInterpolatedFeature(featureGrid, uint2(imageOutput.size(0), imageOutput.size(1)));

    // Run the MLP on the input features.
    //
    mlp.Output output = mlp.eval(feature); 

    // First 3 elements of the output are the RGB color values.
    //
    imageOutput.storeOnce(uint3(dispatchIdx.x, dispatchIdx.y, 0), output.vals[0]);
    imageOutput.storeOnce(uint3(dispatchIdx.x, dispatchIdx.y, 1), output.vals[1]);
    imageOutput.storeOnce(uint3(dispatchIdx.x, dispatchIdx.y, 2), output.vals[2]);
}
