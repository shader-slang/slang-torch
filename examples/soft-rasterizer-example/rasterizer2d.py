import slangtorch
import torch
import numpy as np 
import timeit
import matplotlib.pyplot as plt
from torch.autograd import Function

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

rasterizer2d = slangtorch.loadModule("soft-rasterizer2d.slang", verbose=True)

camera = rasterizer2d.Camera(o=(0.0, 0.0), scale=(1.0, 1.0), frameDim=(1024, 1024))
sigma = 0.02

vertices = torch.tensor([[0.5,-0.5], [-0.5,0.5], [-0.5,-0.5]]).type(torch.float).cuda()
color = torch.tensor([0.8, 0.3, 0.3]).type(torch.float).cuda()

class Rasterizer2d(Function):
    @staticmethod
    def forward(ctx, width, height, camera, sigma, vertices, color):
        output = torch.zeros((width, height, 3), dtype=torch.float).cuda()
        rasterizer2d.rasterize(
            camera=camera,
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

        start = timeit.default_timer()
        
        rasterizer2d.rasterize.bwd(
            camera=camera,
            vertices=(vertices, grad_vertices),
            color=(color, grad_color),
            output=(output, grad_output)
        ).launchRaw(
            blockSize=(16, 16, 1), 
            gridSize=((width + 15)//16, (height + 15)//16, 1)
        )
        
        end = timeit.default_timer()

        print("Backward pass: %f seconds" % (end - start))

        return None, None, None, None, grad_vertices, grad_color

rasterizer = Rasterizer2d()

# Render a simple target image.
targetVertices = torch.tensor([[0.7,-0.3], [-0.3,0.2], [-0.6,-0.6]]).type(torch.float).cuda()
targetColor = torch.tensor([0.3, 0.8, 0.3]).type(torch.float).cuda()
targetImage = rasterizer.apply(1024, 1024, camera, sigma, targetVertices, targetColor)

# Setup our training loop.
learningRate = 5e-3
numIterations = 400

# Initialize our parameters.
vertices = torch.tensor([[0.5,-0.5], [-0.5,0.5], [-0.5,-0.5]]).type(torch.float).cuda()
vertices.requires_grad = True
color = torch.tensor([0.8, 0.3, 0.3]).type(torch.float).cuda()
color.requires_grad = True

# Setup our optimizer.
optimizer = torch.optim.Adam([vertices, color], lr=learningRate)

# Setup plot
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
    outputImage = rasterizer.apply(1024, 1024, camera, sigma, vertices, color)
    outputImage.register_hook(set_grad(outputImage))

    # Compute the loss.
    loss = torch.mean((outputImage - targetImage) ** 2)

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

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, optimize, frames=numIterations, interval=10)
writer = animation.FFMpegWriter(fps=30)
ani.save('rasterizer2d.mp4', writer=writer)
