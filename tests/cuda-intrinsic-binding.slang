
float get_my_const()
{
    __target_switch
    {
    case cuda:
        __requirePrelude("#include \"cuda-intrinsic-test.cuh\"");
        __intrinsic_asm "cu_get_my_const()";
    }
}

// Test that we can use a function that is defined in the prelude
[AutoPyBindCUDA]
[CUDAKernel]
void getConst(TensorView<float> output)
{
    output[0] = get_my_const();
}
