import imported_module;

[CudaKernel]
void mul_kernel(TensorView<float> A, TensorView<float> result)
{
    uint2 location = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).xy;
    result[location] = computeOutputValue(A, location);
}

[TorchEntryPoint]
TorchTensor<float> multiply(TorchTensor<float> A)
{
    var result = TorchTensor<float>.zerosLike(A);
    let blockCount = uint3(1);
    let groupSize = uint3(A.size(0), A.size(1), 1);

    __dispatch_kernel(mul_kernel, blockCount, groupSize)(A, result);
    return result;
}