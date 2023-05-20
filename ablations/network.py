from torch import nn
import torch
from point_ops.pointnet2_ops import pointnet2_utils
from loss_pcc import chamfer_loss_sqrt, chamfer_loss, density_cd
import numpy as np
from torch.nn import functional as F
from pytorch3d.ops.knn import knn_gather, knn_points


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, kmax=30, ms_list=[10, 20], idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_max = knn(x, k=kmax)   # (batch_size, num_points, k)

    if ms_list is not None:
        ms_list.append(kmax)
        ms_feats = []
        for k in range(len(ms_list)):
            idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points
            idx = idx_max[:, :, :ms_list[k]] + idx_base
            idx = idx.view(-1)
            _, num_dims, _ = x.size()
            xx = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #batch_size * num_points * k + range(0, batch_size*num_points)
            feature = xx.view(batch_size*num_points, -1)[idx, :]
            feature = feature.view(batch_size, num_points, ms_list[k], num_dims) 
            xx = xx.view(batch_size, num_points, 1, num_dims).repeat(1, 1, ms_list[k], 1)
            feature = torch.cat((feature-xx, xx), dim=3).permute(0, 3, 1, 2).contiguous()
            ms_feats.append(feature)
        feature = None
        ms_list.pop()
    else:
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points
        idx = idx_max + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, kmax, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, kmax, 1)
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        ms_feats = None
    return feature, ms_feats      # (batch_size, 2*num_dims, num_points, k), list of (batch_size, 2*num_dims, num_points, k_list)


class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat, kmax=40, ms_list=None): # layer1:[20,30], layer2:[30], layer3:None (i.e. no convblk)
        super(ConvBlock, self).__init__()
        self.ms_list = ms_list
        self.k = kmax if ms_list is not None else 20

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_feat), nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_feat), nn.LeakyReLU(negative_slope=0.2))
        self.att_conv = nn.Sequential(nn.Conv2d(out_feat, 1, kernel_size=1, bias=True),
                                      nn.BatchNorm2d(1), nn.LeakyReLU(negative_slope=0.2),
                                      nn.Softmax(dim=-1))

    def forward(self, x):
        x, ms_x = get_graph_feature(x, kmax=self.k, ms_list=self.ms_list)   # (batch_size, 6, num_points) -> (batch_size, 6*2, num_points, k)
        if self.ms_list is not None:
            x1_list = []
            for j in range(len(ms_x)):
                x = ms_x[j]
                x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
                x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
                # x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
                # x1_list.append(x1)
                attn_scores = self.att_conv(x)          # (bs, 64, num_points, k) -> (batch_size, 1, num_points, k)
                x1 = torch.matmul(attn_scores.permute(0,2,1,3), x.permute(0,2,3,1))  # (batch_size, num_points, 1, 64)
                x1_list.append(x1.squeeze().permute(0,2,1))  # (batch_size, 64, num_points)
            x1 = sum(x1_list)/len(x1_list)
        else:
            x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
            x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
            # x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (bs, 64, num_points)
            # x1_list.append(x1)
            attn_scores = self.att_conv(x)          # (bs, 64, num_points, k) -> (batch_size, 1, num_points, k)
            x1 = torch.matmul(attn_scores.permute(0,2,1,3), x.permute(0,2,3,1)).squeeze().permute(0,2,1)  # (batch_size, 64, num_points)
        return x1          


class PCEncoder(nn.Module):
    def __init__(self, kmax=20, **kwargs):
        """
        ms_list: list of k nearest neighbor values for multi-scaling
        kmax: if ms_list is None, k nearest neighbors. if ms_list is not None, the maximum k for multi-scaling
        code_dim: channel dimension of global feature
        """
        super().__init__()
        self.k = kmax
        self.use_nmls = kwargs['use_nmls']
        self.multi_scale = kwargs['multi_scale']
        self.cin = 6 if kwargs['use_nmls'] else 3
        self.code_dim = kwargs['code_dim']

        '''convert layers 1 and 2 to single scale; if conditions to reset the ms_list to None'''
        ms_list = [10,20] if self.multi_scale else None
        # layer 1  
        self.layer1 = ConvBlock(in_feat=self.cin*2, out_feat=64, kmax=30, ms_list=ms_list)
        ms_list = [20] if self.multi_scale else None
        # layer 2
        self.layer2 = ConvBlock(in_feat=64*2, out_feat=128, kmax=30, ms_list=ms_list)

        # layer 3 
        self.layer3 = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=0.2))
        self.att_conv3 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, bias=True),
                                            nn.BatchNorm2d(1), nn.LeakyReLU(negative_slope=0.2),
                                            nn.Softmax(dim=-1))

        # last
        self.last = nn.Sequential(nn.Conv2d(320, self.code_dim, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(self.code_dim), nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x0 = x.permute(0,2,1) # (B 3 N)

        # layer 1
        x1 = self.layer1(x0)

        # layer 2
        x2 = self.layer2(x1)
        #TODO: maybe try a MLP + Residual here for x2

        # layer 3
        x3, _ = get_graph_feature(x2, kmax=self.k, ms_list=None)     # (B, 128, N) -> (B, 128*2, N, k)
        x3 = self.layer3(x3)                                         # (B, 128*2, N, k) -> (B, 128, N, k)
        x3 = x3.max(dim=-1, keepdim=False)[0]                      # (B, 128, N, k) -> (B, 128, N)
        # attn_scores = self.att_conv3(x3)                             # (B, 128, N, k) -> (B, 1, N, k)
        # x3 = torch.matmul(attn_scores.permute(0,2,1,3), x3.permute(0,2,3,1)).squeeze().permute(0,2,1)  # (batch_size, 128, num_points)


        # last bit
        x = torch.cat((x1, x2, x3), dim=1)                           # (B, 320, N) 64+128+128=320


        x = self.last(x.unsqueeze(-1)).squeeze()                     # (B, concat_dim, N) -> (B, emb_dims, N)
        x = x.max(dim=-1, keepdim=False)[0]                           # (B, emb_dims, N) -> (B, emb_dims)

        return x0, x


class TransformerBlock(nn.Module):
    def __init__(self, c_in, c_out, num_heads, ff_dim):
        super().__init__()
        self.to_qk = nn.Conv1d(c_in, c_out*2, kernel_size=1, bias=False)
        self.lnorm1 = nn.LayerNorm(c_out)
        self.mha = nn.MultiheadAttention(c_out, num_heads)
        self.dp_attn = nn.Dropout(0.1) # attn
        self.lnorm2 = nn.LayerNorm(c_out)
        self.ff = nn.Sequential(nn.Linear(c_out, ff_dim),
                                nn.GELU(),
                                nn.Dropout(0.1),
                                nn.Linear(ff_dim, c_out))
        self.dp_ff = nn.Dropout(0.1)

    def forward(self, x):
        Q, K = self.to_qk(x).chunk(2, dim=1) # bcoz convs take [B C N] and linears take [B N C]
        B, C, _ = Q.shape
        Q = self.lnorm1(Q.permute(2, 0, 1))  # 201 bcoz mha takes its qkv tensors as [N B C]
        K = self.lnorm1(K.permute(2, 0, 1))

        mha_out = self.mha(Q, K, K)[0]  # thus: mha takes qkv, but kv here is shared; output[0]: [N B C]. [1]: attn weights per head

        # apply residual of Q to attn w/ dp; coz Q is just and mlp output of x; and dp boost generalization
        Q = self.lnorm2(Q + self.dp_attn(mha_out))
        mha_out = self.ff(Q)
        mha_out = Q + self.dp_ff(mha_out)
        return mha_out.permute(1, 2, 0)  # [B C N]


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
    x = x.permute(0, 2, 1)
    B, N, C = x.shape
    # get indices of cloud1 which has a minimum distance from cloud2
    knn = knn_points(x[:, :, :3], x[:, :, :3], K=k)  # input shd be BNC; dist, k_idx: BNK
    # dist = knn.dists

    grouped_x = knn_gather(x, knn.idx)  

    dot = torch.matmul(grouped_x[:,:,:,3:], x[:,:,3:].view(B, N, 1, 3).permute(0,1,3,2)).squeeze()  #BNK
    cos_sim = dot / (torch.linalg.norm(grouped_x[:,:,:,3:], dim=-1) * torch.linalg.norm(x[:,:,3:], dim=-1).unsqueeze(-1)) #BNK/(BNK * BN1)

    delta_xyz = grouped_x[:,:,:,:3] - x[:,:,:3].view(B, N, 1, 3)
    return cos_sim, torch.cat([grouped_x[:,:,:,:3], delta_xyz], dim=-1)


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
            nn.Conv1d(feat_dims[0], feat_dims[0], kernel_size=1),
            nn.GELU(),
            nn.Conv1d(feat_dims[0], 3, kernel_size=1)  
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

        grouped_deltas = torch.sum(grouped_deltas * cs_weight, dim=2)   # TODO: matmul or mul
    
        #TODO:the sharedmlp blk can be applied on self.conv1 (from network.py) & results torch.mul with grp deltas
        grouped_deltas = self.mlp(grouped_deltas)
        
        return torch.cat([grouped_deltas, fine_out[:, 3:, :]], dim=1)


class PCDecoder(nn.Module):
    def __init__(self, code_dim, scale, rf_level=1, **kwargs):  # num_dense = 16384
        """
        code_dim: dimension of global feature from max operation [B, C, 1]
        num_coarse: number of coarse complete points to generate
        """
        super().__init__()
        self.code_dim = code_dim
        self.num_coarse = 1024 
        self.scale = scale
        self.rf_level = rf_level
        self.cin = 6 if kwargs['use_nmls'] else 3
        self.fps_crsovr = kwargs['fps_crsovr']
        self.mlp = nn.Sequential(
            nn.Linear(self.code_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.code_dim),
            nn.GELU(),
            nn.Linear(self.code_dim, self.cin * self.num_coarse)  # B 1 C*6
        )
        self.seed_mlp = nn.Sequential(nn.Conv1d(self.cin, 64, kernel_size=1),
                                      nn.GELU(),
                                      nn.Conv1d(64, 128, kernel_size=1))
        self.g_feat_mlp = nn.Sequential(nn.Conv1d(self.code_dim, 256, kernel_size=1),  # this will also need adjusting
                                      nn.GELU(),
                                      nn.Conv1d(256, 128, kernel_size=1))
        self.trsf1 = TransformerBlock(c_in=256, c_out=256, num_heads=4, ff_dim=256*2)
        self.trsf2 = TransformerBlock(c_in=256, c_out=256, num_heads=4, ff_dim=256*2)
        self.trsf3 = TransformerBlock(c_in=256, c_out=256*self.scale, num_heads=4, ff_dim=256*2) # this ff_dim shd b x4 0r x8

        """ if rf_level == 2, comment this trsf blk """
        # self.trsf4 = TransformerBlock(c_in=256*self.scale, c_out=128*self.scale, num_heads=4, ff_dim=128*4)
        # self.conv1 = nn.Conv1d(128*self.scale, 128*self.scale, kernel_size=1)

        self.conv1 = nn.Conv1d(256*self.scale, 256*self.scale, kernel_size=1)
        self.last_mlp = nn.Sequential(nn.Conv1d(256+128, 128, kernel_size=1),  # +128
                                      nn.Conv1d(128, 64, kernel_size=1),
                                      nn.GELU(),
                                      nn.Conv1d(64, self.cin, kernel_size=1))

        # self.ppconv = PPConv(c_in=6, feat_dims=[32, 64, 64], k=10)

        if self.rf_level == 2:
            scale2 = 2
            self.seed_mlp1 = nn.Sequential(nn.Conv1d(self.cin, 64, kernel_size=1),
                                           nn.GELU(),
                                           nn.Conv1d(64, 128, kernel_size=1))
            self.g_feat_mlp1 = nn.Sequential(nn.Conv1d(self.code_dim, 256, kernel_size=1),
                                             nn.GELU(),
                                             nn.Conv1d(256, 128, kernel_size=1))
            self.trsf21 = TransformerBlock(c_in=256, c_out=128, num_heads=4, ff_dim=128*2)
            # self.trsf22 = TransformerBlock(c_in=128, c_out=128, num_heads=4, ff_dim=128*2)
            self.trsf23 = TransformerBlock(c_in=128, c_out=128*scale2, num_heads=4, ff_dim=128*2)

            self.conv21 = nn.Conv1d(128*scale2, 128*scale2, kernel_size=1)
            self.last_mlp1 = nn.Sequential(nn.Conv1d(128+128, 128, kernel_size=1),
                                         nn.Conv1d(128, 64, kernel_size=1),
                                         nn.GELU(),
                                         nn.Conv1d(64, self.cin, kernel_size=1))

    def forward(self, partial_pc, glob_feat):  # glob_feat shd be 1024, thus seed of 1024
        partial_pc = partial_pc.permute(0, 2, 1)
        coarse = self.mlp(glob_feat).reshape(-1, self.num_coarse, self.cin)      # [B, num_coarse, 6] coarse point cloud

        if self.fps_crsovr:
            ctrl_x = torch.cat([partial_pc, coarse], dim=1)  # [B, N+num_coarse, 6]
            fps_idx = pointnet2_utils.furthest_point_sample(ctrl_x[:, :, :3].contiguous(), 1024) 
            ctrl_x = ctrl_x.permute(0,2,1).contiguous()
            seed = pointnet2_utils.gather_operation(ctrl_x, fps_idx.int())    # [B, 6, N]
            del ctrl_x
        else:
            seed = coarse.permute(0, 2, 1)  # [B, 6, num_coarse]
        
        torch.cuda.empty_cache()
        
        #since we have to concat the shape code & seed along channel dim, need to make sure dims balance/match
        seed = self.seed_mlp(seed)
        #TODO: PointConv Layer can go here, that is if rf_level=1, else it shd b btwn the two refine steps
        glob_feat1 = self.g_feat_mlp(glob_feat.unsqueeze(-1))
        seed_gf = torch.cat([seed, glob_feat1.repeat(1, 1, seed.size(2))], dim=1)

        seed_gf = self.trsf1(seed_gf)  # [B, c_out, N]
        seed_gf = self.trsf2(seed_gf)  # [B, c_out, N]
        seed_gf = self.trsf3(seed_gf)  # [B, c_out*scale, N]

        # seed_gf = self.trsf4(seed_gf)  # [B, c_out/2 * scale, N]

        B, N, _ = coarse.shape
        seed_gf = self.conv1(seed_gf).reshape(B, -1, N*self.scale)  # [B, c_out/2, N*scale]

        #bcoz seed_gf is potentially upped by self.scale, seed must also be upped to match bf concat & mlp
        seed_gf = torch.cat([seed_gf, seed.repeat(1, 1, self.scale)], dim=1)
        seed_gf_m = self.last_mlp(seed_gf)

        # seed_gf_m = self.ppconv(seed_gf_m)

        if self.rf_level == 2: #we can think of hierrachical completion instead of one time completion like many approaches
            scale2 = 2
            seed1 = self.seed_mlp1(seed_gf_m)
            glob_feat2 = self.g_feat_mlp1(glob_feat.unsqueeze(-1))
            seed_gf = torch.cat([seed1, glob_feat2.repeat(1, 1, seed1.size(2))], dim=1)

            seed_gf = self.trsf21(seed_gf)  # [B, c_out, N]
            # seed_gf = self.trsf22(seed_gf)  # [B, c_out, N]
            seed_gf = self.trsf23(seed_gf)  # [B, c_out*scale, N]
            torch.cuda.empty_cache()
            
            B, C, N = seed1.shape
            seed_gf = self.conv21(seed_gf).reshape(B, -1, N*scale2)  # [B, c_out, N*scale]

            #bcoz seed_gf is potentially upped by self.scale, seed must also be upped to match bf concat & mlp
            seed_gf = torch.cat([seed_gf, seed1.repeat(1, 1, scale2)], dim=1)
            seed_gf = self.last_mlp1(seed_gf)

            return coarse, seed_gf_m, seed_gf  
        else:
            seed_gf = None
            return coarse, seed_gf_m, seed_gf


class PCCNet(nn.Module):
    def __init__(self, kmax=20, code_dim=512, **kwargs):
        super().__init__()
        self.k = kmax
        self.code_dim = code_dim
        self.use_nmls = kwargs['use_nmls']
        self.enc = PCEncoder(kmax=self.k, code_dim=self.code_dim, use_nmls=kwargs['use_nmls'],
                             multi_scale=kwargs['multi_scale'])
        self.dec = PCDecoder(code_dim=code_dim, scale=4, rf_level=1, use_nmls=kwargs['use_nmls'],
                             fps_crsovr=kwargs['fps_crsovr'])

    def forward(self, x):
        p_input, glob_feat = self.enc(x)
        coarse_out, fine_out, finer_out = self.dec(p_input, glob_feat)
        if finer_out is not None:
            return coarse_out, fine_out.permute(0, 2, 1), finer_out.permute(0, 2, 1)
        else:
            return coarse_out, fine_out.permute(0, 2, 1), finer_out


def validate(model, loader, epoch, args, device, rand_save=False): 
    print("Validating ...")
    model.eval()
    num_iters = len(loader)

    with torch.no_grad():
        cdt_coarse, cdp_coarse, cdt_fine, cdp_fine, d_fine, d_coarse = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for i, data in enumerate(loader):
            #data
            xyz = data[0][:, :, :6].to(device).float()  # partial: [B 2048, 6] include normals
            if not model.use_nmls:
                xyz = xyz[:, :, :3]
            #model
            coarse, fine, finer = model(xyz)
            # coarse, fine = coarse[:, :, :3], fine[:, :, :3]
            #losses
            gt_xyz = data[1][:, :, :3].to(device).float()  # partial: [B 16348, 6]
            if args.tr_loss == 'dcd':
                d_fine += density_cd(fine[:, :, :3], gt_xyz).item()  #inputs shd be BNC; cd_p
                d_coarse += density_cd(coarse[:, :, :3], gt_xyz).item() 

            cdp_fine += chamfer_loss_sqrt(fine[:, :, :3], gt_xyz).item()  #inputs shd be BNC; cd_p
            cdp_coarse += chamfer_loss_sqrt(coarse[:, :, :3], gt_xyz).item()  
            cdt_fine += chamfer_loss(fine[:, :, :3], gt_xyz).item()  # cd_t
            cdt_coarse += chamfer_loss(coarse[:, :, :3], gt_xyz).item()  

            if rand_save and args.max_epoch == epoch and i in [0, 7, 15]:
                # if finer is not None:
                #     np.savez(str(args.file_dir) + '/rand_outs.npz', gt_pnts=gt_xyz.data.cpu().numpy(), 
                #                                                     final_pnts=finer.data.cpu().numpy(), 
                #                                                     fine_pnts=fine.data.cpu().numpy(), 
                #                                                     coarse_pnts=coarse.data.cpu().numpy(),
                #                                                     als_pnts=xyz.data.cpu().numpy()[:, :, :3])
                # else:
                np.savez(str(args.file_dir) + f'/rand_outs{i}.npz', gt_pnts=gt_xyz.data.cpu().numpy(),
                                                                final_pnts=fine.data.cpu().numpy(), 
                                                                coarse_pnts=coarse.data.cpu().numpy(),
                                                                als_pnts=xyz.data.cpu().numpy()[:, :, :3])

    return {'fine_d': d_fine/num_iters, 'fine_p': cdp_fine/num_iters, 'coarse_p': cdp_coarse/num_iters, 'coarse_d': d_coarse/num_iters, 'fine_t': cdt_fine/num_iters, 'coarse_t': cdt_coarse/num_iters}
