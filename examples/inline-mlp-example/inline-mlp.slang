// Expect WARP_SIZE, NUM_THREADS_PER_BLOCK to be defined from the python side.

static const int warpSize = WARP_SIZE;
static const int numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
static const int numWarpsPerBlock = NUM_THREADS_PER_BLOCK / WARP_SIZE;

void __inline_matmul_impl(Ptr<float> input, Ptr<float> weights, Ptr<float> output,
                          constexpr int W, constexpr int C, constexpr int O, 
                          constexpr int M, constexpr int N, constexpr int K)
{
    __target_switch
    {
    case cuda:
        // See "cuda_matmul_prelude.cuh" for the tiled matmul implementation.
        __requirePrelude("#include \"cuda_matmul_prelude.cuh\"");
        __intrinsic_asm "wmma_inline_matmul< $3, $4, $5, $6, $7, $8 >($0, $1, $2)";
    }
}

__generic<let W: int, let C : int, let O : int, let M : int, let N : int, let K : int>
void __inline_matmul(
    Ptr<float> input, Ptr<float> weights, Ptr<float> output)
{
    __inline_matmul_impl(input, weights, output, W, C, O, M, N, K);
}
__generic<let C : int> 
struct Feature: IDifferentiable
{
    float vals[C];
}

/*
 * An implementation of a CxC linear layer that is designed to be
 * used 'inline' within a larger kernel. 
 */
struct Linear<let C : int>
{
    typedef Feature<C> Input;
    typedef Feature<C> Output;
    typedef Ptr<float> SharedMemRef;

    DiffTensorView weights;
    DiffTensorView bias;

    uint calcOffset<let N : int>()
    {
        uint3 threadIdx = cudaThreadIdx();
        uint3 blockDim = cudaBlockDim();
        return (((threadIdx.x / warpSize)) + threadIdx.y * (blockDim.x / warpSize)) * N;
    }

    SharedMemRef wtBufferForCurrentWarp()
    {
        // Used for staging both weights & d_weights
        static groupshared float shared_weights_buffer[numWarpsPerBlock * C * C];

        uint warpOffset = calcOffset<C * C>();
        return &shared_weights_buffer[warpOffset];
    }

    SharedMemRef inpBufferForCurrentWarp()
    {
        // Used for staging both inputs & d_inputs
        static groupshared float shared_inputs_buffer[numThreadsPerBlock * C];

        uint warpOffset = calcOffset<warpSize * C>();
        return &shared_inputs_buffer[warpOffset];
    }

    SharedMemRef outBufferForCurrentWarp()
    {
        // Used for staging both outputs & d_outputs
        static groupshared float shared_output_buffer[numThreadsPerBlock * C];

        uint warpOffset = calcOffset<warpSize * C>();
        return &shared_output_buffer[warpOffset];
    }

    SharedMemRef moveWeightsToSharedMem<let colMajor: bool>()
    {
        SharedMemRef wtPtr = wtBufferForCurrentWarp();

        // Copy weights to shared memory.
        uint3 threadIdx = cudaThreadIdx();
        [ForceUnroll]
        for (uint i = 0; i < C; i += warpSize)
        {
            [ForceUnroll]
            for (uint j = 0; j < C; j ++)
            {
                var threadIdInWarp = threadIdx.x % warpSize;
                if ((i + threadIdInWarp) >= C)
                    continue;

                if (colMajor)
                    wtPtr[(i + threadIdInWarp) * C + j] = weights[i + threadIdInWarp, j];
                else
                    wtPtr[j * C + (i + threadIdInWarp)] = weights[i + threadIdInWarp, j];
            }
        }

        return wtPtr;
    }

    SharedMemRef storeArray<let N: int, let colMajor: bool>(Ptr<float> memptr, float input[N])
    {
        uint threadIdInWarp = cudaThreadIdx().x % warpSize;

        // Each thread in the warp will move N contiguous elements to their corresponding shared memory.
        if (!colMajor)
        {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                memptr[threadIdInWarp * N + i] = input[i];
        }
        else
        {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                memptr[i * warpSize + threadIdInWarp] = input[i];
        }

        return memptr;
    }

    void loadArray<let N : int, let colMajor : bool>(Ptr<float> memptr, out float input[N])
    {
        uint threadIdInWarp = cudaThreadIdx().x % warpSize;

        // Each thread in the warp will move N contiguous elements to their corresponding shared memory.
        if (!colMajor)
        {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                input[i] = memptr[threadIdInWarp * N + i];
        }
        else
        {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                input[i] = memptr[i * warpSize + threadIdInWarp];
        }
    }

    SharedMemRef moveInputsToSharedMem<let N: int>(float input[N])
    {
        // Pack in row-major format.
        SharedMemRef inPtr = inpBufferForCurrentWarp();
        return storeArray<N, false>(inPtr, input);
    }

    SharedMemRef moveDInputsToSharedMem<let N : int>(float input[N])
    {
        // Pack in col-major format.
        SharedMemRef inPtr = inpBufferForCurrentWarp();
        return storeArray<N, true>(inPtr, input);
    }

    SharedMemRef moveDOutputsToSharedMem<let N: int>(float d_output[N])
    {
        // Pack in _transposed_ row-major.. which is just col-major.
        SharedMemRef outPtr = outBufferForCurrentWarp();
        return storeArray<N, true>(outPtr, d_output);
    }

    void moveOutputsToLocalArray<let N: int>(out float outputs[N], SharedMemRef bias)
    {
        SharedMemRef outPtr = outBufferForCurrentWarp();
        loadArray<N, false>(outPtr, outputs);
        for (int i = 0; i < N; i++)
            outputs[i] = outputs[i] + bias[i];
    }

    [BackwardDerivative(eval_bwd)]
    Output eval(Input in_feature)
    {
        uint warpOffset = calcOffset<warpSize * C>();

        SharedMemRef inPtr = moveInputsToSharedMem<C>(in_feature.vals);
        SharedMemRef wtPtr = moveWeightsToSharedMem<false>();

        SharedMemRef outPtr = outBufferForCurrentWarp();

        __inline_matmul<warpSize, C, C, 16, 16, 8>(inPtr, wtPtr, outPtr);

        Output out_feature;
        moveOutputsToLocalArray<C>(out_feature.vals, bias.primal.data_ptr());

        return out_feature;
    }

    void eval_bwd(inout DifferentialPair<Input> in_feature_pair, Feature<C>.Differential d_output)
    {
        uint warpOffset = calcOffset<C * C>();

        // Accumulate input derivatives.
        {
            SharedMemRef dOutPtr = moveInputsToSharedMem<C>(d_output.vals);
            SharedMemRef wtPtr = moveWeightsToSharedMem<true>();

            SharedMemRef dInPtr = outBufferForCurrentWarp();

            __inline_matmul<warpSize, C, C, 16, 16, 8>(dOutPtr, wtPtr, dInPtr);

            Input.Differential d_input_feature;
            loadArray<C, false>(dInPtr, d_input_feature.vals);
            in_feature_pair = DifferentialPair<Input>(in_feature_pair.p, d_input_feature);
        }
        
        // Accumulate weight derivatives.
        {
            SharedMemRef inPtr = moveDInputsToSharedMem<C>(in_feature_pair.p.vals);
            SharedMemRef outPtr = moveDOutputsToSharedMem<C>(d_output.vals);

            SharedMemRef wtPtr = wtBufferForCurrentWarp();

            __inline_matmul<C, warpSize, C, 16, 16, 8>(outPtr, inPtr, wtPtr);

            uint3 threadIdx = cudaThreadIdx();
            uint3 blockSize = cudaBlockDim();

            [ForceUnroll]
            for (uint i = 0; i < C; i += warpSize)
            {
                [ForceUnroll]
                for (uint j = 0; j < C; j++)
                {
                    var threadIdInWarp = threadIdx.x % warpSize;
                    if ((i + threadIdInWarp) >= C)
                        continue;
                    float oldVal;
                    weights.diff.diff.InterlockedAdd(
                        uint2(j, i + threadIdInWarp), wtPtr[(i + threadIdInWarp) * C + j], oldVal);
                }
            }
        }

        // Accumulate bias derivatives.
        {
            [ForceUnroll]
            for (int i = 0; i < C; i ++)
            {
                float total_d_bias = WaveActiveSum(d_output.vals[i]);
                if (WaveIsFirstLane())
                {
                    float oldVal;
                    bias.diff.diff.InterlockedAdd(i, total_d_bias, oldVal);
                }
            }
        }
    }
};

struct MLP<let C: int, let N : int>
{
    typedef Feature<C> Input;
    typedef Feature<C> Output;

    Linear<C> layers[N];

    __init(DiffTensorView weights[N], DiffTensorView bias[N])
    {
        [ForceUnroll]
        for (int i = 0; i < N; i++)
        {
            layers[i] = { weights[i], bias[i] };
        }
    }

    [Differentiable]
    Output eval(Input in_feature)
    {
        Output out_feature = in_feature;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
        {
            out_feature = layers[i].eval(out_feature);

            // ReLU
            [ForceUnroll]
            for (int j = 0; j < C; j++)
            {
                out_feature.vals[j] = max(0.0f, out_feature.vals[j]);
            }
        }
        return out_feature;
    }
};

