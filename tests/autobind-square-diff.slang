[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void square(DiffTensorView input, DiffTensorView output)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= input.size(0))
        return;
    
    output[dispatchIdx.x] = input[dispatchIdx.x] * input[dispatchIdx.x];
}