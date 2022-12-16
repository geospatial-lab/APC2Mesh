import torch
from point_ops.Chamfer3D.dist_chamfer_3D import chamfer_3DDist

chamfer_distance = chamfer_3DDist()


def chamfer_loss(p1, p2):
    d1, d2, _, _ = chamfer_distance(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_loss_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_distance(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2