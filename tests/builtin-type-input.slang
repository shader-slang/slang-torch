[AutoPyBindCUDA]
[CUDAKernel]
void plain_copy_float3(float3 input, TensorView<float> output)
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // If the thread index is beyond the input size, exit early.
    if (dispatchIdx.x >= 1)
        return;

    output[0] = input.x;
    output[1] = input.y;
    output[2] = input.z;
}

[AutoPyBindCUDA]
[CUDAKernel]
void plain_copy_float3x3(float3x3 input, TensorView<float> output)
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // If the thread index is beyond the input size, exit early.
    if (dispatchIdx.x >= 1)
        return;

    output[0] = input[0][0];
    output[1] = input[0][1];
    output[2] = input[0][2];
    output[3] = input[1][0];
    output[4] = input[1][1];
    output[5] = input[1][2];
    output[6] = input[2][0];
    output[7] = input[2][1];
    output[8] = input[2][2];
}