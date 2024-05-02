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

struct MyStruct
{
    float3x3 m;
};

[AutoPyBindCUDA]
[CUDAKernel]
void plain_copy_struct(MyStruct input, TensorView<float> output)
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // If the thread index is beyond the input size, exit early.
    if (dispatchIdx.x >= 1)
        return;

    output[0] = input.m[0][0];
    output[1] = input.m[0][1];
    output[2] = input.m[0][2];
    output[3] = input.m[1][0];
    output[4] = input.m[1][1];
    output[5] = input.m[1][2];
    output[6] = input.m[2][0];
    output[7] = input.m[2][1];
    output[8] = input.m[2][2];
}

[AutoPyBindCUDA]
[CUDAKernel]
void plain_copy_array(float3[3] input, TensorView<float> output)
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // If the thread index is beyond the input size, exit early.
    if (dispatchIdx.x >= 1)
        return;

    output[0] = input[0].x;
    output[1] = input[0].y;
    output[2] = input[0].z;
    output[3] = input[1].x;
    output[4] = input[1].y;
    output[5] = input[1].z;
    output[6] = input[2].x;
    output[7] = input[2].y;
    output[8] = input[2].z;
}