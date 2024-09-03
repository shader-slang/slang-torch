static const half kFactor = 2.h;

half computeOutputValue(TensorView<half> A, uint2 loc)
{
    return A[loc] * kFactor;
}

[CudaKernel]
void mul_kernel(TensorView<half> A, TensorView<half> result)
{
    uint2 location = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).xy;
    result[location] = computeOutputValue(A, location);
}

[TorchEntryPoint]
TorchTensor<half> multiply(TorchTensor<half> A)
{
    var result = TorchTensor<half>.zerosLike(A);
    let blockCount = uint3(1);
    let groupSize = uint3(A.size(0), A.size(1), 1);

    __dispatch_kernel(mul_kernel, blockCount, groupSize)(A, result);
    return result;
}