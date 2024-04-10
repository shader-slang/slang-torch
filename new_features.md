

## PyBindCUDA to auto-generate launch kernel.

PyBindCUDA(Auto) will generate a python binding with launch parameters specified at run-time
[PyBindCUDA(Auto)]

In slang we should be able to mark a function with `[PyBindCUDA(Auto)]`
``` C++
[PyBindCUDA(Auto)]
[CudaKernel]
void sqr(TensorView<float> input, TensorView<float> output)
{
   ...
}

// --- Synthesized ----

// Entry point method
void pybind_sqr(uint3 __pybindcuda_blockSize, uint3 __pybindcuda_gridSize, TorchTensor<float> input, TorchTensor<float> output)
{
    __dispatch_kernel(sqr, gridSize, blockSize)(input, output);
}


// Reflection code.
Py::List pybind_sqr_funcinfo()
{
    // Name of the input for (kwargs)
    return Py::Tuple(
        Py::List("__pybindcuda_blockSize", "__pybindcuda_gridSize", "input", "output"),s
        Py::None(),  // Fwd deriv, if any
        Py::None(),  // Bwd deriv, if any
        ); 
}

// Binding code.
PYBIND11("sqr", pybind_sqr);
PYBIND11("sqr_funcinfo", pybind_sqr);

```

On the python size, we will generate a wrapper.
``` Python
class Module:
    def __init__(self):
        self.m = loadModule(path)

    def sqr(self, *args, launchBlockSize=None, launchGridSize=None):
        argnames = self.m.sqr_funcinfo()
        blockSizeArgIndex = argnames.indexOf("__pybindcuda_blockSize")
        gridSizeArgIndex = argnames.indexOf("__pybindcuda_gridSize")
        
        args.insert(blockSizeArgIndex, launchBlockSize)
        args.insert(gridSizeArgIndex, launchGridSize)
        self.m.sqr(blockSize, gridSize, input, output)
```

## Differentiable Tensors

We'll have two diff tensor implementations for commonly used patterns: 
1. `DiffTensorViewAA<T>` that offers four ways to aggregate gradients based on the method used to load (`load`, `loadFixed`, `loadUniform`, `loadUniformOnce`)
2. `DiffTensorViewR<T, N>` that uses `N` replicas of the gradient buffer and aggregates them back up later, during the `.after()` call that the compiler inserts after the Slang kernel is complete.
3. `DiffTensorViewASR<T, N>` that uses an atomic scatter reduce operation in its `.after()` call.

``` C++
struct DiffTensorViewAA<T : IDifferentiable>
{
    TensorView<T> primal;
    TensorView<T> differential; 

    [BackwardDerivative(load_bwd)]
    T load(uint idx)
    {
        return primal[idx];
    }

    void _load_bwd(uint idx, T.Differential dOut)
    {
        differential.InterlockedAdd(idx, dOut);
    }

    // Repeat for other index types: uint2, uint3, uint4 ....

    // All threads load from the same index (the reverse-mode uses wave-activesum)
    [BackwardDerivative(loadFixed_bwd)]
    T loadFixed(uint idx)
    {
        return primal[idx];
    }

    void _loadFixed_bwd(uint idx, T.Differential dOut)
    {
        var dOutSum = WaveActiveSum(dOut);
        if (WaveIsFirstLane())
            differential.InterlockedAdd(idx, dOutSum);
    }

    // All threads load from a unique index, and will never alias
    // (the reverse-mode uses regular '+=')
    // 
    [BackwardDerivative(loadUniform_bwd)]
    T loadUniform(uint idx)
    {
        return primal[idx];
    }

    void _loadUniform_bwd(uint idx, T.Differential dOut)
    {
        differential[idx] = differential[idx] + dOut; 
    }

    // A uniform load that occurs only once from each address
    // for the duration of the program
    // (the reverse-mode uses simple assignment '=')
    // 
    [BackwardDerivative(loadUniformOnce_bwd)]
    T loadUniformOnce(uint idx)
    {
        return primal[idx];
    }

    void _loadUniformOnce_bwd(uint idx, T.Differential dOut)
    {
        differential[idx] = dOut; 
    }

    // Repeat for other index types: uint2, uint3, uint4 ....
};

```

## Differentiable Kernel Binding

We should be able to mark a method with `[PyBindCUDAFwdDiff(Auto)]` and/or `[PyBindCUDABwdDiff(Auto)]` to link our kernel's *derivatives* automatically.
``` C++

[PyBindCUDA(Auto)]
[PyBindCUDAFwdDiff(Auto)]
[PyBindCUDABwdDiff(Auto)]
[Differentiable]
[CudaKernel]
void sqr(DiffTensorViewAA<float> input, DiffTensorViewAA<float> output)
{
   float3 globalIdx = cudaGlobalIdx();
   if (globalIdx.x > input.shape[0])
      return;

   output[globalIdx.x] = input[globalIdx.x] * input[globalIdx.x];
}

// --- Synthesized ----

// Host-side binding for primal
void pybind_sqr(TorchTensor<float> input, TorchTensor<float> output)
{
    var input_tensorview = TensorView<float>.from_tensor(input);
    var output_tensorview = TensorView<float>.from_tensor(output);
    __dispatch_kernel(sqr)(input, output);
}

// Slang method that calls the fwd-deriv.
void sqr_fwd(DiffTensorViewAA<float> input, DiffTensorViewAA<float> output)
{
    fwd_diff(sqr)(input, output);
}

// Host-side method for _fwd (note that tensors have become tuples, the python side
// also expects tuples)
// 
void pybind_sqr_fwd(uint3 __pybindcuda_blockSize, uint3 __pybindcuda_gridSize,
      std::tuple<TorchTensor<float>, TorchTensor<float>> input,
      std::tuple<TorchTensor<float>, TorchTensor<float>> output)
{
    var input_tensorview = DiffTensorViewAA<float>.from_tensor(input[0], input[1]);
    var output_tensorview = DiffTensorViewAA<float>.from_tensor(output[0], output[1]);
    __dispatch_kernel(
         sqr_fwd,
         __pybindcuda_blockSize,
         __pybindcuda_gridSize)(input_tensorview, output_tensorview);
}

// Reflection code.
Py::List pybind_sqr_fwd_funcinfo()
{
    // Name of the input for (kwargs)
    return Py::Tuple(
        Py::List("__pybindcuda_blockSize", "__pybindcuda_gridSize", "input", "output"),
        Py::None(),
        Py::None());
}

// Slang method that calls the bwd-deriv.
void sqr_bwd(DiffTensorViewAA<float> input, DiffTensorViewAA<float> output)
{
    bwd_diff(sqr)(input, output);
}

// Host-side method for _fwd (note that tensors have become tuples, the python side
// also expects tuples)
// 
void pybind_sqr_bwd(uint3 __pybindcuda_blockSize, uint3 __pybindcuda_gridSize,
      std::tuple<TorchTensor<float>, TorchTensor<float>> input,
      std::tuple<TorchTensor<float>, TorchTensor<float>> output)
{
    var input_tensorview = DiffTensorViewAA<float>.from_tensor(input[0], input[1]);
    var output_tensorview = DiffTensorViewAA<float>.from_tensor(output[0], output[1]);
    __dispatch_kernel(
         sqr_bwd,
         __pybindcuda_blockSize,
         __pybindcuda_gridSize)(input_tensorview, output_tensorview);
}

// Reflection code.
Py::List pybind_sqr_bwd_funcinfo()
{
    // Name of the input for (kwargs)
    return Py::Tuple(
        Py::List("__pybindcuda_blockSize", "__pybindcuda_gridSize", "input", "output"),
        Py::None(),
        Py::None());
}

// Binding code.
PYBIND11("sqr_bwd", pybind_sqr_bwd);
PYBIND11("sqr_bwd_funcinfo", pybind_sqr_bwd_funcinfo);

```

We disallow `out` & `inout` modifiers on tensor objects when used with [PyBindCUDA]
Can use other types/struct types (non-tensor), but they will not produce a derivative for now when used with [PyBindCUDA].
Future: we can return the derivatives of arbitrarily-typed inputs as a `namedtuple`, if required.

