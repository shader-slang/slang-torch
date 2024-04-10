[PyExport("Foo")]
struct Foo
{
    TensorView<float> A;
    TensorView<float> B;
};

[AutoPyBindCUDA]
[CudaKernel]
void multiply(Foo foo, TensorView<float> result)
{
    uint2 location = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).xy;
    if (location.x < foo.A.size(0) && location.y < foo.B.size(1))
        result[location] = foo.A[location] * foo.B[location];
}
 