[AutoPyBindCUDA]
[CUDAKernel]
void square(TensorView<float> input, TensorView<float> output)
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // If the thread index is beyond the input size, exit early.
    if (dispatchIdx.x >= input.size(0))
        return;

    output[dispatchIdx.x] = input[dispatchIdx.x] * input[dispatchIdx.x];
}
