## Fit Bezier Curve to Heart Shaped Equation
import torch
import slangpy
import os 
import matplotlib.pyplot as plt

N = 20
c = 2 
m = slangpy.loadModule('bezier.slang', defines={"NUM_CTRL_PTS": N, "DIM":c})


def heart(t):
    t = t*2*torch.pi
    x = 16*(torch.sin(t))**3
    y = 13*torch.cos(t) - 5*torch.cos(2*t) -2*torch.cos(3*t) - torch.cos(4*t)
    return torch.hstack([x.reshape(-1,1),y.reshape(-1,1)])

def ellipse(t, a, b):
    t = t*2*torch.pi
    x = a * (torch.cos(t))
    y = b * (torch.sin(t))
    return torch.hstack([x.reshape(-1,1),y.reshape(-1,1)])

def astrid(t, a):
    t = t*2*torch.pi
    x = a * (torch.cos(t))**3
    y = a * (torch.sin(t))**3
    return torch.hstack([x.reshape(-1,1),y.reshape(-1,1)])

def curve_from_coeffs(t, coeffs):
    """ To check if coefficients are correct """
    output = torch.zeros(t.shape[0], coeffs.shape[1]).cuda()
    for i in range(coeffs.shape[0]):
        output = output + (t**i).view(-1,1) * coeffs[i].view(1,-1)
    return output 

class Bezier2D(torch.autograd.Function):
	@staticmethod
	def forward(ctx, t, control_pts):
		"""
		t: M,1 (torch.tensor) on GPU, parameter for bezier curves
		control_pts: N,2 (torch.tensor) 
		"""
		outputs = torch.zeros(t.shape[0], control_pts.shape[1]).cuda()
		kernel_with_args = m.bezier2D(t=t, control_pts=control_pts, output=outputs)
		NUM_BLOCKS = 1 + t.shape[0] // 1024
		kernel_with_args.launchRaw(
			blockSize=(NUM_BLOCKS, 1, 1),
			gridSize=(1024, 1, 1))
		ctx.save_for_backward(t, control_pts, outputs)
		return outputs

	@staticmethod
	def backward(ctx, grad_outputs):
		(t, control_pts, outputs) = ctx.saved_tensors
		grad_ctrl_pts = torch.zeros_like(control_pts).cuda()
		grad_t  = torch.zeros_like(t).cuda()
  		# Note: When using DiffTensorView, grad_output gets 'consumed' during the reverse-mode.
		# If grad_output may be reused, consider calling grad_output = grad_output.clone()

		kernel_with_args = m.bezier2D.bwd(t=(t, grad_t),
                                                       control_pts=(control_pts, grad_ctrl_pts),
                                                       output=(outputs, grad_outputs))
		NUM_BLOCKS = 1 + t.shape[0] // 1024
		kernel_with_args.launchRaw(
			blockSize=(NUM_BLOCKS, 1, 1),
			gridSize=(1024, 1, 1))

		return grad_t, grad_ctrl_pts




### Initializing Control points, and Target Curve
num_pts = 100
t = torch.linspace(0.0, 1, num_pts, dtype=torch.float).cuda()

savedir =  "./heart_20"
os.makedirs(savedir, exist_ok=True)
# gt_pts = ellipse(t, 3.0, 4.0)
# gt_pts = astrid(t, 3.0)
gt_pts = heart(t)
control_pts = torch.rand((N, 2), dtype=torch.float).cuda()
control_pts.requires_grad_(True)


### Experiment 1 - Learning control points to match heart
# Define a custom parameter, for example, a single value parameter.
opt_param = torch.nn.Parameter(control_pts)
pts = Bezier2D.apply(t, opt_param)

plt.figure()
plt.plot(pts[:,0].detach().cpu().numpy()/0.9, pts[:,1].detach().cpu().numpy()/0.9, color='red',label='predicted')
plt.title('Bezier Curve Initialization')
plt.savefig(os.path.join(savedir, 'init.png'))

# Use an optimizer, for example, SGD, and register the custom parameter with it.
lr_init = 0.01
optimizer = torch.optim.Adam([opt_param], lr=lr_init)

loss_curve = []
for epoch in range(10000):  # Assuming 10000 epochs
    pts = Bezier2D.apply(t,  opt_param)
    loss = ((torch.linalg.norm(pts  - gt_pts, dim=1))).mean()
    optimizer.zero_grad()
    for pg in optimizer.param_groups:
        pg['lr'] = lr_init * 0.99
    loss.backward()
    optimizer.step()
    loss_curve.append(loss.item())
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

plt.figure()
pts = Bezier2D.apply(t,  opt_param)
plt.plot(pts[:,0].detach().cpu().numpy()/0.95, pts[:,1].detach().cpu().numpy()/0.95, color='red',label='predicted')
plt.plot(gt_pts[:,0].detach().cpu().numpy(), gt_pts[:,1].detach().cpu().numpy(), color='green',label='gt')
plt.title('HEART')
plt.legend(['Predicted', 'Ground Truth'])
plt.savefig(os.path.join(savedir, 'control_pts_descent.png'))

plt.figure()
plt.plot(loss_curve)
plt.title('Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss Value')
plt.savefig(os.path.join(savedir, 'Loss_Curve.png'))