static const int N = NUM_CTRL_PTS;
static const int c = DIM;
static const int N1 = c * (N- 1);

struct MatrixG<let C : int> : IDifferentiable
{
    float vals[C];
}

int nCi(int n, int i) 
{
    if (i > n) return 0;
    if (i == 0 || i == n) return 1;
    if (i > n - i) i = n - i;
    
    int result = 1;
    for (int k = 1; k <= i; ++k)
    {
        result *= n - k + 1;
        result /= k;
    }
    
    return result;
}

[PreferRecompute]
int fact(int n) 
{
    int result = 1;
    for (int i = 1; i <= n; ++i)
    {
        result *= i;
    }
    return result;
}

/*
 * We bottleneck the component calculation through a single function tagged as [PreferRecompute]
 * to avoid interediate memory allocations for contents of loops.
 */
[PreferRecompute]
[Differentiable]
float calc_component(uint i, uint j, uint k, DiffTensorView control_pts)
{
    return pow(-1, i + j) * control_pts[i, k] / (fact(i) * fact(j - i));
}

// Function to assemble matrix to compute determinant of to compute SDF. 
[Differentiable]
void asm_mat(uint index, DiffTensorView output, matrix<float, N, c> coeffs)
{
    /** Function to create the matrix whose determinant is to be evaluated to get the sdf
    @param coeffs: Tensor (N,c)
    **/

    for (int i = 0; i < N - 1; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < c; k++)
            {
                output.storeOnce(uint3(index, (k * (N - 1) + i), j + i), coeffs[j][k]);
            }
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void bezier2D(DiffTensorView t, DiffTensorView control_pts, DiffTensorView output)
{
    /** @param t (tensor Mx1) : indices between 0-1 to traverse across the Bezier curve
    **  @param control_pts (Nx2): N - Degree of Bezier Curve 2D
    */
    uint3 tIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // If the thread index is beyond the input size, exit early.
    if (tIdx.x > t.size(0))
        return;
    [ForceUnroll]
    for (int i = 0; i <= N - 1; i++)
    {
        output[tIdx.x, 0] = output[tIdx.x, 0] + nCi(N - 1, i) * pow((1 - t[tIdx.x]), (N - 1 - i)) * pow(t[tIdx.x], i) * control_pts[i, 0];
        output[tIdx.x, 1] = output[tIdx.x, 1] + nCi(N - 1, i) * pow((1 - t[tIdx.x]), (N - 1 - i)) * pow(t[tIdx.x], i) * control_pts[i, 1];
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void bezier2DSDF(DiffTensorView xy, DiffTensorView bcoeffs, DiffTensorView output)
{
    /** @param xy  - M,c
    **  @param bcoeffs - N,c
    **  @return output - M, N1, N1  matrix for each point at which SDF is to be evaluated
    ** Each thread computes the SDF value for a given xy coordinate from the determinant function above. Maybe change it up to be just differentiable, and not AutoPyBindCUDA
    */
    uint3 tIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    matrix<float, N, c> coeffs; // = compute_coeffs_device(control_pts);

    // copying coefficients to a separate variable for each thread.
    for (int i = 0; i < N; i++)
        for (int j = 0; j < c; j++)
            coeffs[i][j] = bcoeffs[i, j];

    int M = xy.size(0); // xy - shaped M,2
    if (tIdx.x > M)
    {
        return;
    }

    float coord[c];
    [ForceUnroll]
    for (int i = 0; i < c; i++)
        coord[i] = xy[tIdx.x, i];

    [ForceUnroll]
    for (int i = 0; i < c; i++)
        coeffs[0][i] -= coord[i];

    asm_mat(tIdx.x, output, coeffs);
}

[AutoPyBindCUDA]
[CudaKernel]
[Differentiable]
void compute_coeffs(DiffTensorView control_pts, DiffTensorView output)
{
    // Compute the coefficients a_i for t^i, for bezier polynomial \sum a_i . t^i
    for (int k = 0; k < c; k++)
    {
        for (int j = 0; j < N; j++)
        {
            int nCj = fact(N - 1) / fact(N - 1 - j); // degree of the polynomial is N-1
            float sum = 0;

            for (int i = 0; i < N; i++)
            {
                if (i <= j)
                    sum += calc_component(i, j, k, control_pts);
            }
            output.storeOnce(uint2(j, k), nCj * sum);
        }
    }
}
