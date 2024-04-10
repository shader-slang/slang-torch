# 1-triangle differentiable rasterizer using Monte Carlo edge sampling in Slang

**NOTE:** This is an **advanced tutorial** that follows-up on the previous two rasterizer tutorials. This tutorial assumes some background on differentiable rendering (see [here](https://diff-render.org/tutorials/cvpr2021/) for an in-depth tutorial)

It is also strongly recommended that you read the previous rasterizer tutorials, as all the examples share a common code structure:

1. [Forward Rasterizer (non-differentiable)](../fwd-rasterizer-example/#readme)
2. [Differentiable Soft Rasterizer](../soft-rasterizer-example/#readme)

Soft rasterization is simple to implement and understand, and provides useful gradients. However, it also **alters** the underlying triangle model by diffusing the edges, and can be unsuitable for many situations where the user may wish to keep the discrete nature of the shapes.

Unfortunately, as the [soft rasterizer](../soft-rasterizer-example/#readme) tutorial discusses, the discrete triangle test function is usually expressed as an `if` condition, and such conditions are non-differentiable.

Can we still provide a correct differentiable model? The next section elaborates on this.

## Monte Carlo Edge-Sampling 
We will employ the *edge-sampling* idea used in the [differentiable vector graphics](https://dl.acm.org/doi/pdf/10.1145/3414685.3417871) & [differentiable Monte-Carlo ray-tracing](https://people.csail.mit.edu/tzumao/diffrt/) papers, which uses the idea of *edge-sampling* to accurately compute the derivative of discontinuous graphics primitives. Though they explore a wide range of vector elements & 3D triangle mesh data-structures, here we'll focus on a simple triangle made up of 3 discontinuous line segment boundaries.

Here's a quick overview of edge-sampling (please refer to the papers for a deep-dive):

### Rasterization as a numerical integration problem.

Fundamentally, rasterization solves a numerical integration problem, by computing the mean color of the geometry covering the 'square' region of a given pixel.

Formally, we can express each pixel's value as an integral:

$$I_{i, j} = \iint_{P} \big(f(x, y; \theta) * c + (1 - f(x,y;\theta)) * b \big) \cdot\mathrm{d}x\cdot\mathrm{d}y $$

where $f(x, y;\mathbf{\theta})$ is an indicator function that is `1` if $(x, y)$ is inside a triangle defined by its vertices $\mathbf{\theta}$ and `0` if not, $c$ is the triangle color & $b$ is the background color
 
### Derivative of the rasterization integral (Reynold's transport theorem)
The derivative of the rasterizer w.r.t some parameter $\theta$, according to the [Reynold's transport theorem](https://en.wikipedia.org/wiki/Reynolds_transport_theorem) is the integral of the derivative over the same domain $P$ plus an integral over all discontinuous boundaries in $P$ (represented by $\partial P$)

$$\partial_\theta I_{i, j} = \iint_{P} \partial_\theta f \cdot (c - b) \cdot\mathrm{d}x\cdot\mathrm{d}y + \int_{\partial P} f(x, y; \theta)\cdot\mathbf{n}\cdot (\partial_\theta x, \partial_\theta y)\cdot\mathrm{d}s $$

were $s$ is a 1D coordinate that maps all discontinuous boundaries in the integral
 
In practice, for our 1-triangle rasterizer, we will estimate the two integrals by sampling the 2D pixel area for the first term & sampling from any triangle edge segments (which will be 1D) that fall into the pixel's bounds.

That is, we will replace the integrals with discrete sums over samples ($N_a$ area samples + $N_e$ edge samples):

$$\widehat{\partial_\theta I_{i, j}} = \frac{1}{N_a} {\sum^{N_a}\_{m=0}} \partial_\theta f (x_m, y_m) \cdot (c - b) +  \frac{1}{N_e} {\sum^{N_e}\_{n=0}} f(x_n, y_n; \theta)\cdot\Big(\mathbf{n} \cdot (\partial_\theta x, \partial_\theta y)\Big) $$

Our next section will express this as a rasterizer which **upon** differentiation gives us the above estimator.

## Implementing a differentiable rasterizer

We'll use the following construction for our rasterizer:

$$\widehat{I_{i, j}} = \frac{1}{N_a} {\sum^{N_a}\_{m=0}} f(x_m, y_m) \cdot (c - b) + \frac{1}{N_e} {\sum^{N_e}\_{n=0}} f(x_n, y_n; \theta)\cdot\mathbf{n}\cdot \Big((x_n, y_n) - (\bar{x}_n, \bar{y}_n)\Big)$$
where $(\bar{x}_n, \bar{y}_n)$ is $(x_n, y_n)$ _detached_ from the graph (equivalent to PyTorch's `stop_gradient`), the first term represents area samples from the 2D domain of the pixel, and the second term represents edge samples from edges (if present) in the pixel.

**Note:** The key idea here is that differentiating this estimator using any auto-diff system (w.r.t $\theta$) will yield the derivative estimator that we discussed in the previous section.

Our implementation is broken down into the following **key** functions in `hard-rasterizer.slang`:

1. `render_pixel()`: Computes the overall rasterization estimate for a given pixel ID (denoted by `pixel`), and the triangle parameters (denoted by `triangle`). Note that it computes the first "interior" term with $N_a=4$ and the second "boundary" term with $N_e=1$
2. `Triangle::shade()`: Computes the value of the triangle's color at the given point $x, y$. This includes testing if the point is inside the triangle and multiplying with the difference between color and background ($f(x_m, y_m)$).
3. `Triangle::sampleFromBoundary()`: Draws a random sample from one of the triangle's edges **if** it overlaps with the given pixel. The implementation calls `sampleFromEdge()` and uses a random roll to decide between the samples if multiple edges overlap.
4. `Triangle::sampleFromEdge()`: Draws a random sample from a single edge, **if** it intersects with the pixel's box. We take advantage of Slang's generics to represent an 'optional' type with the `Maybe<T>` wrapper struct that holds a valid bit.

Our implementation also takes advantage of slangtorch's CUDA auto-binding to write Python-based unit tests for key functions. See `unittest_bindings.slang` for the wrappers & `test.py` for the Python unit tests.

## Running the example

You can run the given code with:
```shell
python rasterizer2d.py
```

Upon running the example, you should see a generated optimization video (`rasterizer2d.mp4`)

Dependencies:
 - `slangtorch` **v1.1.10** or later (`pip install slangtorch`)
 - `torch` **2.0** or later with CUDA **11** (or later). Use the [torch website](https://pytorch.org/) to find the right version

For visualization:
 - `matplotlib`
 - `ffmpeg`

Hardware requirements:
 - CUDA-capable GPU.

