import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt

from image_model import RenderImage

# Load a target image as a torch tensor
target_image = torch.from_numpy(plt.imread('media/jeep.jpg')).cuda()[:, :, :3].contiguous()

# Convert from ByteTensor to FloatTensor & normalize to [0, 1]
target_image = target_image.type(torch.float) / 255.0

# Make three random 16x16 matrices
torch.manual_seed(0)
I = torch.diag(torch.ones(16, dtype=torch.float)).cuda().contiguous()
w1 = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')
w2 = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')
w3 = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')

b1 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')
b2 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')
b3 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')

# Create a feature grid at a lower resolution with random values .
# note that the feature grid size is num_grid_cells + (1, 1), for the far corner features.
#
feature_grid_dims = (32, 32)
feature_grid = torch.randn((feature_grid_dims[0] + 1, feature_grid_dims[1] + 1, 14), 
                            dtype=torch.float, requires_grad=True, device='cuda:0')

# Setup optimization loop
optimizer = torch.optim.Adam([w1, w2, w3, b1, b2, b3, feature_grid], lr=3e-2)
loss_fn = torch.nn.MSELoss()

intermediate_images = []
iterations = 4000

import time
start = time.time()

for i in range(iterations):
    y_pred = RenderImage.apply(
        512, 512,
        feature_grid,
        I + w1 * 0.05,
        I + w2 * 0.05,
        I + w3 * 0.05,
        b1, b2, b3)

    loss = loss_fn(y_pred, target_image)

    print(f"Iteration {i}, Loss: {loss.item()}")

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if i % 20 == 0:
        intermediate_images.append(y_pred.detach().cpu().numpy())

end = time.time()

# Display images side-by-side
import matplotlib.pyplot as plt
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(intermediate_images[0])
ax2.imshow(y_pred.detach().cpu().numpy())
ax3.imshow(target_image.detach().cpu().numpy())

# Label images
ax1.set_title('Initial')
ax2.set_title(f'Optimized ({iterations} iterations in {end - start:.2f} seconds)')
ax3.set_title('Target')

plt.show()

# Save a video.
import cv2
height, width, layers = y_pred.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.mp4', fourcc, 30, (width * 2, height))
for image in intermediate_images:
    image = np.clip(image, 0, 1)
    image = np.concatenate([image, target_image.detach().cpu().numpy()], axis=1)

    # Convert BGR to RGB
    image = image[:, :, ::-1]
    video.write((image * 255).astype(np.uint8))

cv2.destroyAllWindows()
video.release()
