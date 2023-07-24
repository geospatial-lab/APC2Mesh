import torch
import torch.nn as nn
import os
import importlib

chamfer_found = importlib.find_loader("emd_cuda") is not None
if not chamfer_found:
    ## Cool trick from https://github.com/chrdiller
    print("Jitting emd_cuda")

    from torch.utils.cpp_extension import load
    emd_cuda = load(name="emd_cuda",
          sources=[
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["emd.cpp"]),
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["emd_kernel.cu"]),
              ])
    print("Loaded JIT 3D CUDA earth mover distance")

else:
    import emd_cuda
    print("Loaded compiled 3D CUDA earth mover distance")


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


class EarthMoverDistance(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, xyz1, xyz2):
        """
        Args:
            xyz1 (torch.Tensor): (b, N1, 3)
            xyz2 (torch.Tensor): (b, N2, 3)

        Returns:
            cost (torch.Tensor): (b)
        """
        if xyz1.dim() == 2:
            xyz1 = xyz1.unsqueeze(0)
        if xyz2.dim() == 2:
            xyz2 = xyz2.unsqueeze(0)
        cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
        return cost
