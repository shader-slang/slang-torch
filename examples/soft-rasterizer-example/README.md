# 1-triangle differentiable soft rasterizer in Slang

In the [forward rasterizer tutorial](../fwd-rasterizer-example/README.md), we showed how to write a simple 1-triangle non-differentiable rasterizer to produce an image from triangle coordinates.

Here we'll look at using Slang's differentiable programming features to make a differentiable version. Note here that a rasterizer is inherently discontinuous and cannot be trivially differentiated. Instead we will need to use one of the various methods to make our rasterizer differentiable.

Here, we will follow the approach used by the popular [SoftRasterizer](https://github.com/ShichenLiu/SoftRas) paper, albeit for a single triangle to keep things simple.

We will use the forward rasterizer code as a starting point. The final code is available in this same repository [here][../soft-rasterizer-example/]

## Converting the discontinuous `triangle()` to the continuous `softTriangle()`

The core idea of soft rasterization is to turn the boolean triangle test function into a soft test that returns a probability value based on the *distance to the triangle*. 

We will use a straightforward approach to compute the distance to a given triangle (`v1, v2, v3`) from any given point (`uv`), and apply a sigmoid non-linearity to it.

Here's the logic for `distanceToTriangle`:
1. Compute *signed* distances to the three edges using `distanceToEdge()` (-ve 'inwards', +ve 'outwards')
2. If exactly two of three distances are +ve, then use the distance to closest vertex (using `distanceToVertex()`) since one of the vertices must be the closest point on the triangle.
3. Otherwise, use the max of signed distances (one of the edges must contain the closest point)
   
And the logic to compute `distanceToEdge`:
1. If the triangle formed by `(pt, u, v)` is obtuse, use distance to closest vertex.
2. If not, use signed distance to line formed by `(u, v)`

Here is the Slang code that implements this logic:
```csharp
[Differentiable]
float distanceToVertex(float2 xy2, float2 v)
{
    // Compute the distance from a point to a vertex
    return length(xy2 - v);
}

[Differentiable]
float distanceToEdge(float2 u, float2 v, float2 pt)
{
    float2 e = v - u;
    float2 n = float2(-e.y, e.x);
    float2 d = pt - u;
    float n_dot_d = dot(n, d);

    // Compute the signed distance from a point to an edge
    if (dot(pt - u, v - u) < 0.f)
        return distanceToVertex(pt, u) * (sign(n_dot_d)); // u is the closest vertex
    else if (dot(pt - v, u - v) < 0.f)
        return distanceToVertex(pt, v) * (sign(n_dot_d)); // v is the closest vertex
    else
    {
        return n_dot_d / length(n); // closest edge
    }
}

[Differentiable]
float distanceToTriangle(float2 xy, float2 v1, float2 v2, float2 v3)
{
    // Minimum distance to the edge of the triangle
    float d1 = distanceToEdge(v2, v1, xy);
    float d2 = distanceToEdge(v3, v2, xy);
    float d3 = distanceToEdge(v1, v3, xy);

    int sd1 = sign(d1);
    int sd2 = sign(d2);
    int sd3 = sign(d3);

    if (sd1 > 0.f && sd2 > 0.f)
        return distanceToVertex(xy, v2); // v2 is the closest vertex
    else if (sd1 > 0.f && sd3 > 0.f)
        return distanceToVertex(xy, v1); // v1 is the closest vertex
    else if (sd2 > 0.f && sd3 > 0.f)
        return distanceToVertex(xy, v3); // v3 is the closest vertex
    else 
        return max(max(d1, d2), d3);

}
```

Note that we have marked all these methods with the `[Differentiable]` attribute, which allows gradients to propagate through our method.

Now, we can write our `softTriangle` compactly:

```csharp
[Differentiable]
float sigmoid(float x, float sigma)
{
    return 1.0 / (1.0 + exp(-x / sigma));
}

[Differentiable]
float softTriangle(float2 xy, float2 v1, float2 v2, float2 v3, float sigma)
{
    float d = distanceToTriangle(xy, v1, v2, v3);
    return sigmoid(-d, sigma);
}
```

Note the new `sigma` argument to control the blurryness of the sigmoid.

Our new **differentiable** rasterization kernel is now as follows:

```csharp
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void rasterize(
    Camera camera,
    float sigma,
    DiffTensorView vertices,
    DiffTensorView color,
    DiffTensorView output)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x > output.size(0) || globalIdx.y > output.size(1))
        return;

    // Load vertices of our triangle.
    // Assume our input tensor is of the form (3, 2) where 3 is the number of vertices
    // and 2 is the number of coordinates per vertex.
    //
    float2 v1 = float2(vertices[uint2(0, 0)], vertices[uint2(0, 1)]);
    float2 v2 = float2(vertices[uint2(1, 0)], vertices[uint2(1, 1)]);
    float2 v3 = float2(vertices[uint2(2, 0)], vertices[uint2(2, 1)]);
    float3 c = float3(color[0], color[1], color[2]);
    float3 background_color = float3(1.f);

    // Compute result for the current pixel.
    float2 screen_sample = globalIdx.xy + 0.5;
    float2 world_sample = camera.screenToWorld(screen_sample);

    float hit = softTriangle(world_sample, v1, v2, v3, sigma);

    float3 result = hit * c + (1 - hit) * float3(1.f);
    
    // Write-back using the 'storeOnce' method that has a more efficient 
    // derivative implementation if each index is written to only once.
    // 
    output.storeOnce(uint3(globalIdx.xy, 0), result.x);
    output.storeOnce(uint3(globalIdx.xy, 1), result.y);
    output.storeOnce(uint3(globalIdx.xy, 2), result.z);
}
```

We must now use `DiffTensorView` instead of `TensorView<T>` for any tensor that we intend to propagate derivatives to. This is because the latter are not differentiable by default, although they can be used as buffers for non-differentiable data or custom gradient aggregation (in fact `DiffTensorView` is a built-in wrapper around two `TensorView<T>`, using Slang's custom derivatives to provide a set of default differentiable accessors).

Let's try to run our soft rasterizer to see what kind of triangle we get. The 'forward' pass can be invoked almost exactly as we did before in the forward rasterizer example (except for the additional `sigma` parameter)

```python
import torch
import slangtorch

rasterizer2d = slangtorch.loadModule("soft-rasterizer2d.slang", verbose=True)

camera = rasterizer2d.Camera(o=(0.0, 0.0), scale=(1.0, 1.0), frameDim=(1024, 1024))

output = torch.empty((width, height, 3), dtype=torch.float).cuda()
rasterizer2d.rasterize(camera=camera, sigma=0.02, vertices=vertices, color=color, output=output).launchRaw(
    blockSize=(16, 16, 1), 
    gridSize=((width + 15)//16, (height + 15)//16, 1))

import matplotlib.pyplot as plt
plt.imshow(outputImage.permute(1, 0, 2).cpu().numpy(), origin='lower', extent=[-1, 1, -1, 1])
plt.show()
```

You should now see a slightly blurry triangle!

Now, since we marked our kernel `rasterize()` with `[Differentiable]`, `slangtorch` will automatically bind the derivative kernel which can be invoked using `rasterize.bwd()`. 

Let's use that to run a simple optimization. But before that, we will wrap our calls to `rasterize()` & `rasterize.bwd()` in a `torch.autograd.Function` to make it easy to invoke from our optimization loop. 

```python
import slangtorch
import torch

from torch.autograd import Function

rasterizer2d = slangtorch.loadModule("soft-rasterizer2d.slang", verbose=True)

class Rasterizer2d(Function):
    @staticmethod
    def forward(ctx, width, height, camera, sigma, vertices, color):
        output = torch.zeros((width, height, 3), dtype=torch.float).cuda()
        rasterizer2d.rasterize(
            camera=camera,
            sigma=sigma,
            vertices=vertices,
            color=color,
            output=output
        ).launchRaw(
            blockSize=(16, 16, 1), 
            gridSize=((width + 15)//16, (height + 15)//16, 1)
        )

        ctx.camera = camera
        ctx.sigma = sigma
        ctx.save_for_backward(vertices, color, output)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        vertices, color, output = ctx.saved_tensors 
        camera = ctx.camera
        sigma = ctx.sigma

        grad_vertices = torch.zeros_like(vertices)
        grad_color = torch.zeros_like(color)
        grad_output = grad_output.contiguous()

        width, height = grad_output.shape[:2]

        rasterizer2d.rasterize.bwd(
            camera=camera,
            sigma=sigma,
            vertices=(vertices, grad_vertices),
            color=(color, grad_color),
            output=(output, grad_output)
        ).launchRaw(
            blockSize=(16, 16, 1), 
            gridSize=((width + 15)//16, (height + 15)//16, 1)
        )

        return None, None, None, None, grad_vertices, grad_color

return Rasterizer2d()

```

Our torch function takes as input the image size, the camera parameters, sigma and the vertex & color buffers. Note that since we only intend to differentiate w.r.t vertices & colors, our `.backward()` only needs to return non-None values for these tensors. 

Now, we can use this torch function in an optimization loop:

```python

rasterizer = Rasterizer2d()

camera = rasterizer2d.Camera(o=(0.0, 0.0), scale=(1.0, 1.0), frameDim=(1024, 1024))
sigma = 0.02

# Render a 'target' image: a 'green' triangle with perturbed vertices.
targetVertices = torch.tensor([[0.7,-0.3], [-0.3,0.2], [-0.6,-0.6]]).type(torch.float).cuda()
targetColor = torch.tensor([0.3, 0.8, 0.3]).type(torch.float).cuda()

targetImage = rasterizer.apply(1024, 1024, camera, sigma, targetVertices, targetColor)

# Setup our training parameters.
learningRate = 5e-3
numIterations = 400

# Initialize our parameters.
vertices = torch.tensor([[0.5,-0.5], [-0.5,0.5], [-0.5,-0.5]], requires_grad=True).type(torch.float).cuda()
color = torch.tensor([0.8, 0.3, 0.3], requires_grad=True).type(torch.float).cuda()

# Setup our optimizer.
optimizer = torch.optim.Adam([vertices, color], lr=learningRate)

# Setup plot
fig = plt.figure()

# Run our training loop.
for i in range(numIterations):
    print("Iteration %d" % i)

    # Forward pass: render the image.
    outputImage = rasterizer.apply(1024, 1024, camera, sigma, vertices, color)

    # Compute the MSE loss.
    loss = torch.mean((outputImage - targetImage) ** 2)

    # Backward pass: compute the gradients.
    loss.backward()

    # Update the parameters.
    optimizer.step()

    # Zero the gradients.
    optimizer.zero_grad()

# Plot the final image
finalImage = rasterizer.apply(1024, 1024, camera, sigma, vertices, color)
plt.imshow(finalImage.permute(1, 0, 2).detach().cpu().numpy(), origin='lower', extent=[-1, 1, -1, 1])
plt.show()

# Print the recovered values
print('Recovered vertices: ', vertices.detach().cpu().numpy())
print('Recovered color: ', color.detach().cpu().numpy())
```

You should see the following optimized image: 

<img src="rasterizer2d-target.png" width="300">

and see that the recovered values are very close to the true values:

```
Recovered vertices:  [
 [ 0.7 -0.3]
 [-0.3  0.2]
 [-0.6 -0.6]]
Recovered color:  [0.3       0.8000005 0.3      ]
```

The [code for this example](../soft-rasterizer-example/) also includes some matplotlib helpers to plot out an animated version so you can see the optimization progress. 
