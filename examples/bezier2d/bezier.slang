static const int N = NUM_CTRL_PTS;
static const int c = DIM;
static const int N1 = c * (N- 1);

__generic<let C : int> 
struct MatrixG : IDifferentiable
{
    float vals[C];
}

int nCi(int n, int i) {
    if (i > n) return 0;
    if (i == 0 || i == n) return 1;
    if (i > n - i) i = n - i;
    
    int result = 1;
    for (int k = 1; k <= i; ++k) {
        result *= n - k + 1;
        result /= k;
    }
    
    return result;
}

int fact(int n) {
    int result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= i;
    }
    return result;
}

[CudaDeviceExport]
[Differentiable]
matrix<float, N, c> compute_coeffs_device(DiffTensorView control_pts) {
    // Compute the coefficients a_i for t^i, for bezier polynomial \sum a_i . t^i
    matrix<float, N, c> coeffs;
    [ForceUnroll]
    for (int k = 0; k < c; k++) {
        [ForceUnroll]
        for (int j = 0; j < N; j++) {
            int nCj = fact(N - 1) / fact(N - 1 - j); // degree of the polynomial is N-1
            float sum = 0;
            [ForceUnroll]
            for (int i = 0; i < N; i++) {
                if (i <= j) {
                    sum += pow(-1, i + j) * control_pts[i, k] / (fact(i) * fact(j - i));
                }
            }
            coeffs[j][k] = nCj * sum;
        }
    }
    return coeffs;
}


// Function to assemble matrix to compute determinant of to compute SDF. 
[CudaDeviceExport]
[Differentiable]
 MatrixG<N1 * N1> asm_mat(matrix<float, N,c> coeffs) {
    // Function to create the matrix whose determinant is to be evaluated to get the sdf
    // coeffs: Tensor (N,c)
    MatrixG<N1 * N1> mat;

    // Initializing
    [ForceUnroll]
    for (int i = 0; i < N1 * N1; i++)
        mat.vals[i] = 0.0;

    [ForceUnroll]
    for (int i = 0; i < N - 1; i++)
        [ForceUnroll]
        for (int j = 0; j < N; j++)
            [ForceUnroll]
            for (int k = 0; k < c; k++)
                {
                mat.vals[(k * (N - 1) + i) * N1 + j + i] = coeffs[j][k];
            }
    return mat;
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void bezier2D(DiffTensorView t, DiffTensorView control_pts, DiffTensorView output)
{
    // t (tensor Mx1) : indices between 0-1 to traverse across the Bezier curve
    // control_pts (Nx2): N - Degree of Bezier Curve 2D
    // Get the 'global' index of this thread.
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
void bezier2DSDF(DiffTensorView xy, DiffTensorView control_pts, DiffTensorView output) {
    // xy - M,c
    // coeffs - N,c
    // output - M,N1,N1 - matrix for each point at which SDF is to be evaluated
    // coeffs - ,c
    // Each thread computes the SDF value for a given xy coordinate from the determinant function above. Maybe change it up to be just differentiable, and not AutoPyBindCUDA
    uint3 tIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    matrix<float, N, c> coeffs = compute_coeffs_device(control_pts);
    
    int M = xy.size(0); // xy - shaped M,2
    if (tIdx.x > M) {
        return;
    }

    float coord[c];
    [ForceUnroll]
    for (int i = 0; i < c; i++)
        coord[i] = xy[tIdx.x, i];

    [ForceUnroll]
    for (int i = 0; i < c; i++)
        coeffs[0][i] -= coord[i];

    MatrixG<N1 * N1> mat;
    mat = asm_mat(coeffs);
    [ForceUnroll]
    for (int i = 0; i < N1; i++)
        [ForceUnroll]
        for (int j = 0; j < N1; j++)
            output[tIdx.x, i, j] = mat.vals[i*N1 + j];
}

[AutoPyBindCUDA]
[CudaKernel]
[Differentiable]
void compute_coeffs(DiffTensorView control_pts, DiffTensorView output) {
    // Compute the coefficients a_i for t^i, for bezier polynomial \sum a_i . t^i
    matrix<float, N, c> coeffs = compute_coeffs_device(control_pts);
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        [ForceUnroll]
        for (int j = 0; j < c; j++)
            output[i, j] = coeffs[i][j];
}
