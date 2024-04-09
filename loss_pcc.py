import torch
import torch.nn as nn
from point_ops.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
# from point_ops.earth_movers_distance.emd import EarthMoverDistance
from pytorch3d.ops.knn import knn_gather, knn_points
from pytictoc import TicToc

chamfer_distance = chamfer_3DDist()
# emd = EarthMoverDistance()

def chamfer_loss(p1, p2):
    '''cd_t, L2 version of cd_p'''
    d1, d2, _, _ = chamfer_distance(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_loss_sqrt(p1, p2):
    '''cd_p, L1 version of cd_t'''
    d1, d2, _, _ = chamfer_distance(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


# def emd_loss(p1, p2, reduction='mean'):
#     '''
#     earth mover distance loss.
#     Args:
#         p1 (torch.tensor): [B, N, 6]
#         p2 (torch.tensor): [B, N, 6]
#     '''
#     dists = emd(p1, p2)
#     if reduction == 'mean':
#         return torch.mean(dists)
#     elif reduction == 'sum':
#         return torch.sum(dists)


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


def density_cd(x, gt, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    loss = (loss1 + loss2) / 2

    # res = [loss, cd_p, cd_t]
    # if return_raw:
    #     res.extend([dist1, dist2, idx1, idx2])

    return torch.mean(loss)

def calc_cd(pred, gt, return_raw=False, normalize=False, separate=False):

    dist1, dist2, idx1, idx2 = chamfer_distance(gt, pred)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]

    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res


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


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
        thresh (float, optional): stop threshold for computed err. Default: 1e-9
            
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, thresh, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.thresh = thresh # Stopping criterion for Sinkhorn iterations

    def forward(self, x, y):
        tt = TicToc() #create instance of class

        # send x and y to a separate device
        x = x.to('cuda:1')
        y = y.to('cuda:1')
        
        # The Sinkhorn algorithm takes as input three variables :
        D = self._dist_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights [B, M], [B, N]
        wx = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(x.device)
        wy = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(x.device)  
        wy *= (wx.shape[1] / wy.shape[1])

        sum_wx = wx.sum().item()
        sum_wy = wy.sum().item()
        if abs(sum_wx - sum_wy) > 1e-5:
            raise ValueError(
                'sum of marginal weights must match: {} != {}, abs difference = {}'.format(sum_wx, sum_wy, abs(sum_wx - sum_wy)))
        
        log_wx = torch.log(wx) # [B, M]
        log_wy = torch.log(wy) # [B, N]

        # Initialize the iteration with the change of variable
        u = torch.zeros_like(wx)
        v = self.eps * log_wy

        # [B, 1, N] + [B, M, N] = [B, M, N]
        # u_i = u.unsqueeze(2) # [B, M, 1]
        # v_j = v.unsqueeze(1) # [B, 1, N]

        # To check if algorithm terminates via threshold or max_iters reached
        actual_nits = 0

        # Sinkhorn iterations
        tt.tic()
        for i in range(self.max_iter):
            u_prev = u  # useful for checking the update
            v_prev = v

            C_u = (-D + v.unsqueeze(1)) / self.eps  # modified cost for logarithmic updates
            u = self.eps * (log_wx - C_u.logsumexp(dim=2))

            C_v = (-D + u.unsqueeze(2)) / self.eps
            v = self.eps * (log_wy - C_v.logsumexp(dim=1))
            
            max_err_u = torch.max(torch.abs(u_prev - u), dim=1)[0]
            max_err_v = torch.max(torch.abs(v_prev - v), dim=1)[0]

            actual_nits += 1
            if max_err_u.mean() < self.thresh and max_err_v.mean() < self.thresh:
                break
        tt.toc('Sinkhorn iterations')

        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp((D + u.unsqueeze(2) + v.unsqueeze(1)) / self.eps)

        # approx_corr_1 = pi.argmax(dim=1).squeeze(-1)
        # approx_corr_2 = pi.argmax(dim=0).squeeze(-1)

        # Sinkhorn distance
        if u.shape[0] > v.shape[0]:
            cost = (pi * D).sum(dim=2).sum(1)
        else:
            cost = torch.sum(pi * D, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, D

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _dist_matrix(x, y, p=2):
        """
        Returns the matrix of $|x_i-y_j|^p$.
        x: [B, N, D]
        y: [B, M, D]
        """
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if p == 1:
            D = ((x_col - y_lin) ** p).abs().sum(dim=-1) # [B, M, N]
        elif p == 2:
            D = ((x_col - y_lin) ** p).sum(dim=-1) ** (1.0 / p) # [B, M, N]
        return D

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1