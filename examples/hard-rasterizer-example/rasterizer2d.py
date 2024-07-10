import slangtorch
import torch
import numpy as np 
import timeit
import os
import matplotlib.pyplot as plt
from torch.autograd import Function
import torch.nn.functional as F
import sys
import matplotlib.animation as animation
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

vertices = torch.tensor([[0.5,-0.5], [-0.5,0.5], [-0.5,-0.5]]).type(torch.float).cuda()
color = torch.tensor([0.8, 0.3, 0.3]).type(torch.float).cuda()

def setup_rasterizer():
    rasterizer2d = slangtorch.loadModule("hard-rasterizer2d.slang")

    class Rasterizer2d(Function):
        @staticmethod
        def forward(ctx, width, height, vertices, color, rng_state=None):
            if rng_state is None:
                rng_state = torch.randint(-2**31, 2**31, (width, height), dtype=torch.int32).cuda()

            output = torch.zeros((width, height, 3), dtype=torch.float).cuda()

            kernel_with_args = rasterizer2d.rasterize(vertices=vertices, color=color, output=output, rng_state=rng_state)
            kernel_with_args.launchRaw(blockSize=(16, 16, 1), gridSize=((width + 15)//16, (height + 15)//16, 1))
            
            ctx.save_for_backward(vertices, color, output, rng_state)
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            vertices, color, output, rng_state = ctx.saved_tensors 

            grad_vertices = torch.zeros_like(vertices)
            grad_color = torch.zeros_like(color)

            width, height = grad_output.shape[0], grad_output.shape[1]
            grad_output = grad_output.contiguous()

            start = timeit.default_timer()

            kernel_with_args = rasterizer2d.rasterize.bwd(
                vertices=(vertices, grad_vertices),
                color=(color, grad_color),
                output=(output, grad_output),
                rng_state=rng_state)
            kernel_with_args.launchRaw(blockSize=(16, 16, 1), gridSize=((width + 15)//16, (height + 15)//16, 1))

            end = timeit.default_timer()

            print("Backward pass: %f seconds" % (end - start))

            return None, None, grad_vertices, grad_color
    
    return Rasterizer2d()

def pyramid_loss(outputImage, targetImage, levels=6):
    # Iteratively scale down the target image and compute the loss.
    loss = 0.0
    for i in range(levels):
        # Compute the loss.
        level_weight = (2 ** i)
        loss += torch.mean((outputImage - targetImage) ** 2) * level_weight

        # Scale down the target image.
        targetImage = F.avg_pool2d(targetImage, kernel_size=2, stride=2)

        # Scale down the output image.
        outputImage = F.avg_pool2d(outputImage, kernel_size=2, stride=2)

    return loss / levels

rasterizer = setup_rasterizer()

# Render a 'target' image: a green triangle with perturbed vertices.
targetVertices = torch.tensor([[0.7,-0.3], [-0.3,0.2], [-0.6,-0.6]]).type(torch.float).cuda()
targetColor = torch.tensor([0.3, 0.8, 0.3]).type(torch.float).cuda()
targetImage = rasterizer.apply(1024, 1024, targetVertices, targetColor)

# Setup our training loop.
learningRate = 5e-3
numIterations = 400

# Initialize our parameters.
vertices = torch.tensor([[0.1,-0.1], [-0.1,0.1], [-0.1,-0.1]]).type(torch.float).cuda() + 1e-5
vertices.requires_grad = True
color = torch.tensor([0.8, 0.3, 0.3]).type(torch.float).cuda()
color.requires_grad = True

# Setup our optimizer.
optimizer = torch.optim.Adam([vertices, color], lr=learningRate)

# Convert to channels form.
targetImage_ch = targetImage.permute(2, 0, 1)[None, :, :, :]

fig = plt.figure()

ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

# Run our training loop.
def optimize(i):
    print("Iteration %d" % i)

    # Forward pass: render the image.
    outputImage = rasterizer.apply(1024, 1024, vertices, color)
    outputImage.register_hook(set_grad(outputImage))

    # Convert to channels form.
    outputImage_ch = outputImage.permute(2, 0, 1)[None, :, :, :]

    # Compute an image pyramid loss.
    loss = pyramid_loss(outputImage_ch, targetImage_ch, levels=4)

    # Backward pass: compute the gradients.
    loss.backward()

    # Update the parameters.
    optimizer.step()
    
    if i % 10 == 0:
        ax1.clear()
        ax1.imshow(outputImage.permute(1, 0, 2).detach().cpu().numpy(), origin='lower', extent=[-1, 1, -1, 1])
        ax2.clear()
        ax2.imshow(outputImage.grad[:,:,1].T.detach().cpu().numpy(), origin='lower', extent=[-1, 1, -1, 1])
        ax3.clear()
        ax3.imshow(targetImage.permute(1, 0, 2).detach().cpu().numpy(), origin='lower', extent=[-1, 1, -1, 1])

    # Zero the gradients.
    optimizer.zero_grad()


ani = animation.FuncAnimation(fig, optimize, frames=numIterations, interval=10)
writer = animation.FFMpegWriter(fps=30) 
ani.save('rasterizer2d.mp4', writer=writer)
