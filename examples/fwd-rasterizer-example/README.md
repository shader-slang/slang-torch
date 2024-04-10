# Writing a 2D triangle rasterizer in Slang.

This example will demonstrate how to write a simple 2D triangle rasterizer in the Slang shading language, and incorporate it into a PyTorch pipeline using the `slangtorch` python package.

All files discussed in this tutorial are available in this directory [here](../fwd-rasterizer-example/)


## Setup

You will need to install the following packages:
1. `slangtorch`
2. `torch` with CUDA support

Then, we'll create a new `.slang` file to write our rasterizer. The IDE you use doesn't particularly matter, but if you use Visual Studio Code, the [Slang plugin](https://marketplace.visualstudio.com/items?itemName=shader-slang.slang-language-extension) provides Intellisense and interactive code-completion.


## Rendering a 2D triangle onto an image.

Our algorithm is fairly simple, and will serve to highlight the ease of writing graphics programs in *Slang*'s `SIMT` model (as opposed to *PyTorch*'s `NDArray` model)

A quick overview: 
1. Test the center point of the pixel against the three *half-planes* described by the three sides of the triangle. 
2. If the point is in all 3 half-planes, then we'll set that pixel to the given color.
3. If not, then use the background color.
4. Repeat this test for each pixel.

In the single-instruction-multiple-threads (SIMT) programming model, we can simply write logic for each pixel, and launch a thread for each pixel on the GPU. Note that this is in contrast to the `NDArray` model, where we describe logic for *all* pixels (as a tensor).

Here's the test function `triangle()` in Slang:

```csharp
// xy: 2D test position
// v1: vertex position 1
// v2: vertex position 2
// v3: vertex position 3
//
bool triangle(float2 xy, float2 v1, float2 v2, float2 v3)
{
    // 2D triangle test (return 1.0 if the triangle contains the point (x,y), 0.0 otherwise).
    float2 e0 = v2 - v1;
    float2 e1 = v3 - v2;
    float2 e2 = v1 - v3;

    float2 k0 = float2(-e0.y, e0.x);
    float2 k1 = float2(-e1.y, e1.x);
    float2 k2 = float2(-e2.y, e2.x);

    float d0 = dot(k0, xy - v1);
    float d1 = dot(k1, xy - v2);
    float d2 = dot(k2, xy - v3);

    // Check the three half-plane values.
    if(d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0)
        return true;
    else
        return false;
}
```

Note here that, as a shading language, Slang provides built-in vector types such as `float2`, `float3` & `float4`, which can be convenient for 3D graphics coordinate operations. In this case, we leverage them to represent 2D points. All built-in operations & functions defined on scalar types also apply to their vector versions.

Now let's look at the kernel which will invoke `triangle()` on each pixel center value:

```csharp
[AutoPyBindCUDA]
[CUDAKernel]
void rasterize(
    TensorView<float> vertices,
    TensorView<float> color,
    TensorView<float3> output)
{
    uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (dispatch_id.x > output.size(0) || dispatch_id.y > output.size(1))
        return;

    // Load vertices of our triangle.
    // Assume our input tensor is of the form (3, 2) where 3 is the number of vertices
    // and 2 is the number of coordinates per vertex.
    // 
    float2 v1 = float2(vertices[0, 0], vertices[0, 1]);
    float2 v2 = float2(vertices[1, 0], vertices[1, 1]);
    float2 v3 = float2(vertices[2, 0], vertices[2, 1]);
    float3 c = float3(color[0], color[1], color[2]);

    // Convert our 2D thread indices directly into pixel coordinates
    // This way pixel at location, say (9, 20) covers the real-number space 
    // between (9.0, 20.0) to (10.0, 21.0)
    // 
    float2 pixel_coord = dispatch_id.xy;

    // Use white as the default background color.
    float3 background_color = float3(1.0, 1.0, 1.0);

    // Center of the pixel will be offset by 50% of the pixel size,
    // which we will assume is 1 unit by 1 unit.
    // 
    float2 sample_coord = pixel_coord + 0.5;

    bool hit = triangle(sample_coord, v1, v2, v3);

    // If we hit the triangle return the provided color, otherwise 0.
    float3 result = hit ? c : background_color;

    // Fill in the corresponding location in the output image.
    output[dispatch_id.xy] = result;
}
```

Our kernel `rasterize()` begins similar to how CUDA kernels are written, only instead of buffer pointers as input, we use Slang's `TensorView<T>` objects that are inter-compatible with `PyTorch` tensors. 

Note that, the logic in `rasterize()` will be run in parallel on all pixels at once, and the main difference between each thread is its thread `index` which can be computed using `cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();`

Here, we use the index to figure out which pixel the current invocation is handling, and invoke our triangle test.

Marking the `rasterize()` method with `[CUDAKernel]` informs Slang that the method should be compiled to CUDA. `[AutoPyBindCUDA]` is a convenience feature where the Slang compiler & the `slangtorch` package work together to let you invoke your CUDA kernel directly from Python using PyTorch tensors to pass data.

```python
import torch
import slangtorch

rasterizer2d = slangtorch.loadModule("rasterizer2d.slang")

# Describe a red right-angled triangle in the middle of the screen.
vertices = torch.tensor([[768.0, 256.0], [256.0, 768.0], [256.0, 256.0]], dtype=torch.float).cuda()
color = torch.tensor([0.8, 0.3, 0.3], dtype=torch.float).cuda()

# Allocate a tensor for the output image. 
outputImage = torch.empty((1024, 1024, 3), dtype=torch.float).cuda()

# Run our kernel 1024x1024 times by launching a 64x64 grid of blocks
# with 16x16 threads per block.
#
rasterizer2d.rasterize(
    vertices=vertices,
    color=color,
    output=outputImage
).launchRaw(
    blockSize=(16, 16, 1),
    gridSize=(64, 64, 1)
)

# Display our image (swap the axes & show origin on bottom-right)
import matplotlib.pyplot as plt
plt.imshow(outputImage.permute(1, 0, 2).cpu().numpy(), origin='lower')
plt.show()

```
Hopefully, this example served as a good starting point for writing your own rendering kernels. 

However, everything we've discussed so far is non-differentiable.
The repository contains additional examples that discuss building **differentiable** rasterizers in Slang :
1. 1-Triangle Soft Rasterizer using Edge Smoothing ([link](../soft-rasterizer-example/README.md))
2. 1-Triangle 'Hard' Rasterizer using Monte Carlo Edge Sampling ([link](../hard-rasterizer-example/README.md))

The rest of this tutorial will extend the current non-differentiable rasterizer by adding a 2D camera model to prepare it for the differentiable version.
## Extending the simple example with a 2D "Camera" model

The previous example is a minimal rasterizer, but it is rather unweildy to use since all coordinates need to be specified in *screen* space i.e. the pixel at `200, 200` always maps to `200.0, 200.0`. We can make this more elegant by introducing a camera function that translates between *screen* space and *world* space.

We'll take advantage of Slang's object-oriented programming model by defining a `Camera` struct:

```csharp
struct Camera
{
    // World origin
    float2 o;

    // World scale
    float2 scale;

    // Frame dimensions (i.e. image resolution)
    float2 frameDim;

    // Convert from 
    // screen coordinates [(0, 0), (W, H)] to 
    // world coordinates [(o.x - scale.x, o.y - scale.y), (o.x + scale.x, o.y + scale.y)]
    // 
    float2 screenToWorld(float2 uv)
    {
        float2 xy = uv / frameDim;
        float2 ndc = xy * 2.0f - 1.0f;
        return ndc * scale + o;
    }
};
```

We can now simply include a `Camera` parameter in our kernel function, and use it accordingly:

```csharp

[AutoPyBindCUDA]
[CUDAKernel]
void rasterize(
    Camera camera,
    TensorView<float> vertices,
    TensorView<float> color,
    TensorView<float3> output)
{
    uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (dispatch_id.x > output.size(0) || dispatch_id.y > output.size(1))
        return;

    float2 v1 = float2(vertices[0, 0], vertices[0, 1]);
    float2 v2 = float2(vertices[1, 0], vertices[1, 1]);
    float2 v3 = float2(vertices[2, 0], vertices[2, 1]);
    float3 c = float3(color[0], color[1], color[2]);
    float3 background_color = float3(1.0, 1.0, 1.0);

    float2 pixel_coord = dispatch_id.xy;
    float2 screen_sample = pixel_coord + 0.5;

    // Convert screen space pixel coordinates to world space.
    float2 world_sample = camera.screenToWorld(screen_sample);

    bool hit = triangle(world_sample, v1, v2, v3);
    float3 result = hit ? c : background_color;
    output[dispatch_id.xy] = result;
}

```

On the python side, `Camera` will be exposed as a `namedtuple` object with all the struct field names & types. Note that the member _functions_ will not be available by default (unless marked by `[AutoPyBindCUDA]`)

```python
import torch
import slangtorch

rasterizer2d = slangtorch.loadModule("rasterizer2d.slang")

# Create a camera object. Note that since we used scalar fields, the arguments
# must also be simple scalars (and not torch tensors)
# 
# This specific camera instance maps [(0, 0), (1024, 1024)] in image space to [(-1, -1), (1, 1)] in 
# world space.
#
camera = rasterizer2d.Camera(o=(0.0, 0.0), scale=(1.0, 1.0), frameDim=(1024, 1024))

# We can use world space coordinates now!
vertices = torch.tensor([[0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]], dtype=torch.float).cuda()

color = torch.tensor([0.8, 0.3, 0.3], dtype=torch.float).cuda()
outputImage = torch.zeros((1024, 1024, 3), dtype=torch.float).cuda()

rasterizer2d.rasterize(
    camera=camera,
    vertices=vertices,
    color=color,
    output=outputImage
).launchRaw(
    blockSize=(16, 16, 1),
    gridSize=(64, 64, 1)
)

import matplotlib.pyplot as plt
plt.imshow(outputImage.permute(1, 0, 2).cpu().numpy(), origin='lower')
plt.show()

```

That's it! `struct` declarations are automatically made available as `namedtuple` objects if they are used as a parameter type for a method with the `[AutoPyBindCUDA]` attribute. Read more [here](https://shader-slang.com/slang/user-guide/a1-02-slangpy.html#type-marshalling-between-slang-and-python)

