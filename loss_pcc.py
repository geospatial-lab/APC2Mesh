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


def computeCandidateNormals(gt, predicted, k=20):
    pred = predicted.detach().clone()
    B, N, C = gt.shape
    _, S, _ = pred.shape

    # get indices of gt which has a minimum distance from pred
    knn = knn_points(pred[:, :, :3], gt[:, :, :3], K=k) # input shd be B N/S C; dist, k_idx: BSK
    query_gt_grps = knn_gather(gt, knn.idx)[:,:,:,3:]  # BSKC 
        
    query_grp_cnters = query_gt_grps[:, :, 0, :]  # i.e., the smallest dist to the pred query

    # cosine similarity
    dot = torch.matmul(query_gt_grps, query_grp_cnters.view(B, S, 1, 3).permute(0,1,3,2)).squeeze()  #BSK
    cos_sim = dot / (torch.linalg.norm(query_gt_grps, dim=-1) * torch.linalg.norm(query_grp_cnters, dim=-1).unsqueeze(-1)) #BSK/(BSK * BS1)

    # cos_sim = torch.where(cos_sim < 0.75, cos_sim*0.0, cos_sim)  #BSK, C is one so squeezed
    trunc_idx = torch.where(cos_sim < 0.75)
    query_gt_grps[trunc_idx[0], trunc_idx[1], trunc_idx[2], :] = float('nan') # : or 3: 'Nan' bcoz normals parallel to the z-axiz will hv a 0.0 z-value
    query_grp_normal = torch.nanmean(query_gt_grps, axis=2) #BN3 non-zero mean

    return query_grp_normal, query_grp_cnters


def computePlaneWithNormalAndPoint(grp_nmls, query_pnts):
    '''a, b, c are normals which corresponds to the last 3 columns of cloud'''

    a = grp_nmls[:,:,0]  #BN
    b = grp_nmls[:,:,1]
    c = grp_nmls[:,:,2]
    d = torch.diagonal(torch.matmul(query_pnts, grp_nmls.permute(0, 2, 1)), offset=0, dim1=1, dim2=2) * -1.0  # Multiplication by -1 preserves the sign (+) of D on the LHS
    # d = np.diag(np.dot(test_nbr[:,:3], test_nbr[:,3:].T)) * -1.0  # Multiplication by -1 preserves the sign (+) of D on the LHS
    normalizationFactor = torch.sqrt((a * a) + (b * b) + (c * c))

    # if normalizationFactor == 0:
    #     return None
    a /= normalizationFactor
    b /= normalizationFactor
    c /= normalizationFactor
    d /= normalizationFactor

    return (a, b, c, d)


def p2p_dist(gt, pred):
    grp_normals, query_pnts = computeCandidateNormals(gt, pred)
    pl_coeffs = computePlaneWithNormalAndPoint(grp_normals, query_pnts)
    dist = torch.abs(pl_coeffs[0]*pred[:,:,0] + pl_coeffs[1]*pred[:,:,1] + pl_coeffs[2]*pred[:,:,2] + pl_coeffs[3])
    nml_len = torch.sqrt((pl_coeffs[0] * pl_coeffs[0]) + (pl_coeffs[1] * pl_coeffs[1]) + (pl_coeffs[2] * pl_coeffs[2]))
    return torch.mean(dist/nml_len)