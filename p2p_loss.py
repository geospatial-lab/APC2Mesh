import random, torch
from dataset_pcc import CustomDataset
import numpy as np
from pytorch3d.ops.knn import knn_gather, knn_points
from pytictoc import TicToc

def get_data_without_normals(fpath):
    npz_data = np.load(fpath)
    fine = npz_data['final_pnts']
    # gt = npz_data['gt_pnts']
    del npz_data
    return torch.from_numpy(fine).float() #, torch.from_numpy(gt).float()


def get_gt_with_normals(device):
    from torch.utils import data 

    bs = 8  # batch_size
    npoints = 2048

    seed_value = 42
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

    tr_dataset = CustomDataset(split='train', npoints=npoints, device=device)
    tr_loader = data.DataLoader(tr_dataset, batch_size=bs, shuffle=True)

    ts_dataset = CustomDataset(split='test', npoints=npoints, device=device)
    ts_loader = data.DataLoader(ts_dataset, batch_size=bs, shuffle=False)

    for i, data in enumerate(ts_loader):
        #gt data
        gt_xyz = data[1][:, :, :6].to(device).float()  # partial: [B 16348, 6]
        break

    return gt_xyz


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


def p2p_dist(pl_coeffs, points):
    dist = torch.abs(pl_coeffs[0]*points[:,:,0] + pl_coeffs[1]*points[:,:,1] + pl_coeffs[2]*points[:,:,2] + pl_coeffs[3])
    nml_len = torch.sqrt((pl_coeffs[0] * pl_coeffs[0]) + (pl_coeffs[1] * pl_coeffs[1]) + (pl_coeffs[2] * pl_coeffs[2]))
    return dist/nml_len

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fpath = '/outputs/experiments/2023-01-17_22-29/rand_outs.npz'
pred = get_data_without_normals(fpath)  
gt = get_gt_with_normals(device)
t = TicToc()
t.tic() #Start timer
grp_normals, query_pnts = computeCandidateNormals(gt, pred.to(device).float())
plane_coeffs = computePlaneWithNormalAndPoint(grp_normals, query_pnts)
dist = p2p_dist(plane_coeffs, pred.to(device).float())
t.toc()