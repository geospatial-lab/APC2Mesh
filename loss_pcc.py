import torch
from point_ops.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from pytorch3d.ops.knn import knn_gather, knn_points

chamfer_distance = chamfer_3DDist()


def chamfer_loss(p1, p2):
    '''cd_t'''
    d1, d2, _, _ = chamfer_distance(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_loss_sqrt(p1, p2):
    '''cd_p'''
    d1, d2, _, _ = chamfer_distance(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2

def l2_normal_loss(ref, query):
    """
        Parameters
        ----------
        ref: B * N * 6
        query: B * N * 6
        
        Returns
        ----------
        l2 normal loss for points which are closer in points1 & points2
    """
    B, N, C = ref.shape
    
    # get indices of cloud1 which has a minimum distance from cloud2
    knn = knn_points(ref[:, :, :3], query[:, :, :3], K=1)  # input shd be BNC; dist, k_idx: BNK
    # dist = knn.dists
    
    k_points = knn_gather(query, knn.idx)  # BNKC
    
    #dist = torch.mean(torch.sum((points1.view(batch, num_points, 1, channels)[:, :, :, 0:3] - k_points[:, :, :, 0:3]) ** 2, dim=-1))
    normal_loss = torch.mean(torch.sum((ref.view(B, N, 1, C)[:, :, :, 3:6] - k_points[:, :, :, 3:6]) ** 2, dim=-1))
    return normal_loss #, dist


def nbrhood_uniformity_loss(xyz, k_point, k_normal):
    """
        Parameters
        ----------
        points: B * N * 6
        k_point: number of neighbour for point regularization
        k_normal: number of neighbour for normal regularization
        
        Returns
        ----------
        cosine_normal_loss
        normal_neighbor_loss
        point_neighbor_loss
    """
    k = max(k_point, k_normal)
    k = k + 1  # a point also includes itself in knn search
    B, N, C = xyz.shape

    knbrs = knn_points(xyz[:, :, :3], xyz[:, :, :3], K=k)
    # dist = knn.dists
    
    kpnts = knn_gather(xyz, knbrs.idx)

    """Points"""
    pnt_dist = kpnts[:, :, 0, :].view(B, N, 1, C)[:, :, :, :3] - kpnts[:, :, 0:k_point+1, :3]  # x_query - x_nbrs
    
    # remove first column of each row as it is the same point from where min distance is calculated
    point_neighbor_loss = torch.mean(torch.sum(pnt_dist[:, :, 1:, :] ** 2, dim=-1))

    """Normals"""
    n_dist = kpnts[:, :, 0, :].view(B, N, 1, C)[:, :, :, 3:6] - kpnts[:, :, 0:k_normal+1, 3:6]

    # remove first column of each row as it is the same point from where min distance is calculated
    normal_neighbor_loss = torch.mean(torch.sum(n_dist[:, :, 1:, :] ** 2, dim=-1))

    # dot_product = f.normalize(pnt_dist, p=2, dim=-1) * f.normalize(kpnts[:, :, 0, :].view(B, N, 1, C)[:, :, :, 3:6], p=2, dim=-1)
    # cosine_normal_loss = torch.mean(torch.abs(torch.sum(dot_product, dim=-1)))

    return point_neighbor_loss, normal_neighbor_loss  #cosine_normal_loss