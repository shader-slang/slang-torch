# Slang-Torch Package

This package allows you to use Slang as a language to write
PyTorch kernels.

Slang user guide: https://shader-slang.com/slang/user-guide/

`slangtorch` documentation: https://shader-slang.org/slang/docs/user-guide/a1-02-slangpy

## Installation Instructions
Note: Running `slangtorch` requires a CUDA-capable GPU.

- Install the **CUDA Toolkit v12.0 or later**
- Install **PyTorch v2.x** with CUDA support from [https://pytorch.org/](https://pytorch.org/).
- `pip install slangtorch` to install slangtorch.

## Building Wheels Locally
To build the `slangtorch` package locally, run the `build-package-local.sh` script.
This will pull the latest Slang binaries and create a wheel for both Linux and Windows.