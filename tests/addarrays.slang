float computeOutputValue(TensorView<float> A, TensorView<float> B, uint2 loc)
{
    return A[loc] + B[loc];
}

[CudaKernel]
void add_fwd_kernel(TensorView<float> A, TensorView<float> B, TensorView<float> result)
{
    uint2 location = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).xy;
    result[location] = computeOutputValue(A, B, location);
}

[TorchEntryPoint]
TorchTensor<float> add_fwd(TorchTensor<float> A, TorchTensor<float> B)
{
    var result = TorchTensor<float>.zerosLike(A);
    let blockCount = uint3(1);
    let groupSize = uint3(A.size(0), A.size(1), 1);

    __dispatch_kernel(add_fwd_kernel, blockCount, groupSize)(A, B, result);
    return result;
}