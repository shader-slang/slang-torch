import torch
import slangtorch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
