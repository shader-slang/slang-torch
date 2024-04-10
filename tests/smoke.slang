//DISABLE_TEST:SIMPLE: -target cuda -line-directive-mode none

// Verify that we can output a cuda device function with [CudaDeviceExport].
// Disabled until we have FileCheck.

struct ReturnValue
{
    TorchTensor<float> v0;
    TorchTensor<float> v1;
}

struct Inputs
{
    TorchTensor<float> input;
    float value;
}

[CudaKernel]
void myKernel(TensorView<float> inValues, TensorView<float> outValues, float v)
{
    var inputVal = inValues[cudaThreadIdx().xy];
    outValues[cudaThreadIdx().xy] = 0;
    float oldVal;
    outValues.InterlockedAdd(cudaThreadIdx().xy, v + inputVal, oldVal);
}

[TorchEntryPoint]
public __extern_cpp ReturnValue runCompute(Inputs inputs)
{
    let inValues = inputs.input;
    var outValues = TorchTensor<float>.zerosLike(inValues);
    __dispatch_kernel(myKernel, uint3(1, 1, 1), uint3(inValues.size(0), inValues.size(1), 1))(inValues, outValues, inputs.value);
    ReturnValue retVal;
    retVal.v0 = inValues;
    retVal.v1 = outValues;
    return retVal;
}