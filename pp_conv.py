'''Plane-Point refinement module
using cosine similarity of point normals as weighting criterion
thinking weighting is suppose to happen at the nbrhood level 
unlike pointconv that computed the mean of all points to all other points
'''

import torch
from pytorch3d.ops.knn import knn_gather, knn_points
from torch import nn
from torch.nn import functional as F


def cosine_similarity(x, k):
    """
        Parameters
        ----------
        x: input cloud [B N 6]
        
        Returns
        ----------
        cosine similarity of query points to their k-neighborhood points [B N K]
        NB: [-1 to 1] weight space shifted to [0 to 1] in CSBlk
    """
    B, N, C = x.shape
    # get indices of cloud1 which has a minimum distance from cloud2
    knn = knn_points(x[:, :, :3], x[:, :, :3], K=k)  # input shd be BNC; dist, k_idx: BNK
    # dist = knn.dists

    grouped_x = knn_gather(x, knn.idx)  

    dot = torch.matmul(grouped_x[:,:,:,3:], x[:,:,3:].view(B, N, 1, 3).permute(0,1,3,2)).squeeze()  #BNK
    cos_sim = dot / (torch.linalg.norm(grouped_x[:,:,:,3:], dim=-1) * torch.linalg.norm(x[:,:,3:], dim=-1).unsqueeze(-1)) #BNK/(BNK * BN1)

    delta_xyz = grouped_x[:,:,:,:3] - x[:,:,:3].view(B, N, 1, 3)
    return cos_sim, delta_xyz


class CSBlock(nn.Module):
    '''cosine similarity block'''
    def __init__(self, c_in, feat_dims):
        super(CSBlock, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = c_in  
        for c_out in feat_dims:  # [32, 32, 64]
            self.mlp_convs.append(nn.Conv2d(last_channel, c_out, 1))
            self.mlp_bns.append(nn.BatchNorm2d(c_out))
            last_channel = c_out

    def forward(self, cos_sim):
        # cos_sim: shd b [BCKN], so permute(0, 2, 1).unsqueeze(1) coz cos_sim is [BNK]
        cos_sim = cos_sim.permute(0, 2, 1).unsqueeze(1)
        B, C, K, N = cos_sim.shape
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            cos_sim = bn(conv(cos_sim))  
            if i == len(self.mlp_convs)-1: # this takes care of the -ve cos_sim vals (chk)
                cos_sim = torch.sigmoid(cos_sim)
            else:
                cos_sim = F.gelu(cos_sim)

        return cos_sim


class SharedMLPBlock(nn.Module):
    def __init__(self, c_in, feat_dims):
        super(SharedMLPBlock, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = c_in  
        for c_out in feat_dims:  # [32, 32, 64]
            self.mlp_convs.append(nn.Conv2d(last_channel, c_out, 1))
            self.mlp_bns.append(nn.BatchNorm2d(c_out))
            last_channel = c_out

    def forward(self, grouped_xyz):
        # grouped_xyz: shd b [BCKN]
        B, C, K, N = grouped_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_xyz = F.gelu(bn(conv(grouped_xyz))) 

        return grouped_xyz
        

class PPConv(nn.Module):
    '''Point-Plane refinement module'''
    def __init__(self, c_in, feat_dims, k):
        super(PPConv, self).__init__()

        self.k = k
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = c_in  
        for c_out in feat_dims:  # [32, 64, 64]
            self.mlp_convs.append(nn.Conv2d(last_channel, c_out, 1))
            self.mlp_bns.append(nn.BatchNorm2d(c_out))
            last_channel = c_out

        # self.smlp = SharedMLPBlock(c_in=3, feat_dims=[32, 64, 64])
        self.csblk = CSBlock(c_in=1, feat_dims=[32,64])
        self.mlp = nn.Sequential(
            nn.Conv1d(feat_dims[2], feat_dims[0], kernel_size=1),
            nn.GELU(),
            nn.Conv1d(feat_dims[0], feat_dims[0], kernel_size=1),
            nn.GELU(),
            nn.Conv1d(feat_dims[0], 6, kernel_size=1)  
        )

    def forward(self, fine_out):
        """fine_out: shd hv channel of 6, 3 coords 3 normals"""

        grouped_cs, grouped_deltas = cosine_similarity(fine_out, k=self.k)  # takes in both xyz & normals; out [BNK, BNKC] 
        grouped_deltas = grouped_deltas.permute(0,3,2,1)# for conv, grouped_xyz has to be BCKN 

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_deltas = F.gelu(bn(conv(grouped_deltas))) 

        #TODO: (1) no max-scaling, (2) max-scaling w/ sigmoid (3) max-scaling w/ softmax
        max_cs = grouped_cs.max(dim = 2, keepdim=True)[0]
        cs_weight = grouped_cs / max_cs
        cs_weight = self.csblk(cs_weight)  # [BCKN]

        grouped_deltas = torch.sum(grouped_deltas * cs_weight, dim=-2)   # TODO: matmul or mul
    
        #TODO:the sharedmlp blk can be applied on self.conv1 (from network.py) & results torch.mul with grp deltas
        grouped_deltas = self.mlp(grouped_deltas)
        
        return grouped_deltas


x = torch.rand(2, 20, 6) * 2 - 1  #BN3
net = PPConv(c_in=3, feat_dims=[32, 64, 64], k=10)
output = net(x)
print('done !')