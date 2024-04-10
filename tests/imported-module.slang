float computeOutputValue(TensorView<float> A, uint2 loc)
{
    return A[loc] * %FACTOR%;
}
