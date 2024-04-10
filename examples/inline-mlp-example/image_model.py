import slangtorch
import torch
import os

launchBlockSize = (32, 8, 1)

# Before loading our module, we're going to pre-process the TORCH_CUDA_ARCH_LIST tag from the environment. 
# This is to get around a bug with how torch constructs arch flags: explicitly providing arch flags seems to
# not override torch's default arch flags and cause compiler errors due to setting the SM version too low.
# 
# We therefore replace the env list with a restricted list of arches that we know will work.
#
if "TORCH_CUDA_ARCH_LIST" in os.environ:
    # Go through space-separated list of arches and remove any that are below 8.0
    arches = os.environ["TORCH_CUDA_ARCH_LIST"].split(" ")
    arches = [arch for arch in arches if not int(arch.split(".")[0]) <= 7]
    os.environ["TORCH_CUDA_ARCH_LIST"] = " ".join(arches)

m = slangtorch.loadModule('image-model.slang', 
                       defines={
                            'NUM_THREADS_PER_BLOCK': launchBlockSize[0] * launchBlockSize[1] * launchBlockSize[2],
                            'WARP_SIZE': 32})

class RenderImage(torch.autograd.Function):
    def forward(ctx, width, height, feature_grid, *args):
        weights = args[0: 3]
        biases = args[3: 6]
        output = torch.zeros((width, height, 3), dtype=torch.float).cuda()
        
        linear_layers = [m.Linear(weights=weights[i], bias=biases[i]) for i in range(3)]
        mlp = m.MLP(layers=linear_layers)

        blockSize = launchBlockSize
        gridSize = ((width + blockSize[0] - 1) // blockSize[0], (height + blockSize[1] - 1) // blockSize[1], 1)

        m.renderImage(mlp=mlp, featureGrid=feature_grid, imageOutput=output).launchRaw(blockSize=blockSize, gridSize=gridSize)

        ctx.save_for_backward(output, feature_grid, *args)

        return output
    
    def backward(ctx, grad_output):
        output, feature_grid, *args = ctx.saved_tensors
        weights = args[0: 3]
        biases = args[3: 6]

        weights_d = [torch.zeros_like(w) for w in weights]
        biases_d = [torch.zeros_like(b) for b in biases]
        feature_grid_d = torch.zeros_like(feature_grid)

        width, height, _ = output.shape
        
        linear_layers = [m.Linear(weights=(weights[i], weights_d[i]), bias=(biases[i], biases_d[i])) for i in range(3)]
        mlp = m.MLP(layers=linear_layers)

        blockSize = launchBlockSize
        gridSize = ((width + blockSize[0] - 1) // blockSize[0], (height + blockSize[1] - 1) // blockSize[1], 1)

        m.renderImage.bwd(mlp=mlp, featureGrid=(feature_grid, feature_grid_d), imageOutput=(output, grad_output)).launchRaw(blockSize=blockSize, gridSize=gridSize)

        return None, None, feature_grid_d, *weights_d, *biases_d
