import torch
import slangpy
import numpy as np
import matplotlib.pyplot as plt 

N = 6
c = 2 
m = slangpy.loadModule('bezier.slang', defines={"NUM_CTRL_PTS": N, "DIM":c})

class BezierSDF_mats(torch.autograd.Function):
	@staticmethod
	def forward(ctx, xy, control_pts):
		"""
		xy: M,2 torch tensor on GPU, points at which SDF is to be evaluated 
		control_pts: 
		"""
		# coeffs = torch.zeros_like(control_pts, dtype=torch.float).cuda()
		# m.compute_coeffs(control_pts=control_pts, output=coeffs).launchRaw(blockSize=(1, 1, 1), gridSize=(1, 1, 1))
		sdf_mats = torch.zeros(xy.shape[0], c*(N-1), c*(N-1)).cuda()
		kernel_with_args = m.bezier2DSDF(xy=xy, control_pts=control_pts, output=sdf_mats)
		NUM_BLOCKS = 1 + xy.shape[0] // 1024
		kernel_with_args.launchRaw(
			blockSize=(NUM_BLOCKS, 1, 1),
			gridSize=(1024, 1, 1))
		ctx.save_for_backward(xy, control_pts, sdf_mats)
		return sdf_mats

	@staticmethod
	def backward(ctx, grad_sdf_mats):
		(xy, control_pts, sdf_mats) = ctx.saved_tensors
		grad_ctrl_pts = torch.zeros_like(control_pts)
		grad_xy  = torch.zeros_like(xy)
  		# Note: When using DiffTensorView, grad_output gets 'consumed' during the reverse-mode.
		# If grad_output may be reused, consider calling grad_output = grad_output.clone()

		kernel_with_args = m.bezier2DSDF.bwd(xy=(xy, grad_xy),
                                                       control_pts=(control_pts, grad_ctrl_pts),
                                                       output=(sdf_mats, grad_sdf_mats))
		NUM_BLOCKS = 1 + xy.shape[0] // 1024
		kernel_with_args.launchRaw(
			blockSize=(NUM_BLOCKS, 1, 1),
			gridSize=(1024, 1, 1))

		return grad_xy, grad_ctrl_pts


def compute_sdf(control_pts, num_pts, xrange=[0.0,1.0], yrange=[0.0,1.0]):
	px = torch.linspace(xrange[0], xrange[1], num_pts)
	py = torch.linspace(yrange[0], yrange[1], num_pts)

	# Create the meshgrid
	x, y = torch.meshgrid(px, py.flip(dims=[0]), indexing='ij')  # 'i
	xy = torch.stack([x,y], dim=-1 ).view(-1,2).cuda()
	xy.requires_grad_(True)
	sdf_mats = BezierSDF_mats.apply(xy, control_pts)
	sdf = torch.linalg.det(sdf_mats)
	sdf = torch.sign(sdf) * torch.sqrt(torch.abs(sdf))

	return sdf    

def compute_sdf_pts(control_pts, xy):
	""" Compute sdf for a pre-specified point / array of points
	Args:
		xy: (M,2)
		control_pts: (N,2)
	Returns:
		sdf (M,1)
	"""
	sdf_mats = BezierSDF_mats.apply(xy, control_pts)
	sdf = torch.linalg.det(sdf_mats)
	sdf = torch.sign(sdf) * torch.sqrt(torch.abs(sdf))
	return sdf       


def curve_from_coeffs(t, coeffs):
    """ To check if coefficients are correct """
    output = torch.zeros(t.shape[0], coeffs.shape[1]).cuda()
    for i in range(coeffs.shape[0]):
        output = output + (t**i).view(-1,1) * coeffs[i].view(1,-1)
    return output 


## Problem setup. Bezier Curve control points, and an initialization. 
gt_control_pts = torch.tensor([[0.0, 0.0],[0.5, 0.5], [0.0, 1.0], [0.8, 0.5],  [1.0, 1.0], [1.0, 0.0]]).cuda()
gt_control_pts.requires_grad_(True)
init_x, init_y = 0.3, 0.2 #initial location of the point 
xy = torch.tensor([init_x, init_y]).view(-1,2).cuda()
xy.requires_grad_(True)


### Experiment 1 - Gradient Descenting along SDF 
# Define a custom parameter, for example, a single value parameter.
loc_param = torch.nn.Parameter(xy)

# Use an optimizer, for example, SGD, and register the custom parameter with it.
lr_init = 0.00001
optimizer = torch.optim.Adam([loc_param], lr=lr_init)
loc_traj = []
for epoch in range(1000):  # Assuming 1000 epochs
    sdf = compute_sdf_pts(gt_control_pts, loc_param)
    loss = torch.abs(torch.mean(sdf))
    
    optimizer.zero_grad()
    for pg in optimizer.param_groups:
        pg['lr'] = lr_init * loss.item()
    
    loss.backward()
	
    optimizer.step()
    loc_traj.append(loc_param[0].detach().cpu().numpy())
    print(f"Epoch {epoch+1}, Loss: {loss.item()}, Parameter: {[loc_param[0][0].item(), loc_param[0][1].item()]}")



### Plotting
num_pts = 1000
t = torch.linspace(0.0, 1, num_pts, dtype=torch.float).cuda()
coeffs = torch.zeros((N,c), dtype=torch.float).cuda()
m.compute_coeffs(control_pts=gt_control_pts, output=coeffs).launchRaw(blockSize=(4, 1, 1), gridSize=(1, 1, 1))

curve_coeffs = curve_from_coeffs(t, coeffs)

plt.figure()
plt.plot(curve_coeffs[:,0].detach().cpu().numpy(), curve_coeffs[:,1].detach().cpu().numpy())
for i in range(gt_control_pts.shape[0]):
    plt.scatter(gt_control_pts[i][0].detach().cpu(), gt_control_pts[i][1].detach().cpu())


loc_traj = np.array(loc_traj)
plt.scatter(init_x, init_y)
plt.text(init_x, init_y, 'Initialization', rotation=45)

plt.plot(loc_traj[:,0], loc_traj[:,1])
plt.scatter(loc_traj[-1,0], loc_traj[-1,1])
plt.text(loc_traj[-1][0], loc_traj[-1][1], 'After Optimization', rotation=45)

plt.title(f'Gradient Descenting Along SDF')
plt.savefig(f'sdf_descent_{N}pts.png')
