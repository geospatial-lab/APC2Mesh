from torch import nn
import torch
from point_ops.pointnet2_ops import pointnet2_utils
from loss_pcc import chamfer_loss_sqrt, chamfer_loss
import numpy as np

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
            xx = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
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
        self.k = kmax

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_feat), nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_feat), nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x, ms_x = get_graph_feature(x, kmax=self.k, ms_list=self.ms_list)   # (batch_size, 6, num_points) -> (batch_size, 6*2, num_points, k)
        if self.ms_list is not None:
            x1_list = []
            for j in range(len(ms_x)):
                x = ms_x[j]
                x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
                x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
                x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
                x1_list.append(x1)
            x1 = sum(x1_list)/len(x1_list)
        else:
            x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
            x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
            x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        return x1

class PCEncoder(nn.Module):
    def __init__(self, kmax=20, code_dim=512):
        """
        ms_list: list of k nearest neighbor values for multi-scaling
        kmax: if ms_list is None, k nearest neighbors. if ms_list is not None, the maximum k for multi-scaling
        code_dim: channel dimension of global feature
        """
        super().__init__()
        self.k = kmax
        self.code_dim = code_dim

        # layer 1  
        self.layer1 = ConvBlock(in_feat=6*2, out_feat=64, kmax=30, ms_list=[10,20])

        # layer 2
        self.layer2 = ConvBlock(in_feat=64*2, out_feat=128, kmax=20, ms_list=[10])

        # layer 3 
        self.layer3 = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=0.2))

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
        x3 = x3.max(dim=-1, keepdim=False)[0]                        # (B, 128, N, k) -> (B, 128, N)

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

class PCDecoder(nn.Module):
    def __init__(self, code_dim, scale, rf_level=1):  # num_dense = 16384
        """
        code_dim: dimension of global feature from max operation [B, C, 1]
        num_coarse: number of coarse complete points to generate
        """
        super().__init__()
        self.code_dim = code_dim
        self.num_coarse = 1024 
        self.scale = scale
        self.rf_level = rf_level
        self.mlp = nn.Sequential(
            nn.Linear(self.code_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.code_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.code_dim, 6 * self.num_coarse)  # B 1 C*6
        )
        self.seed_mlp = nn.Sequential(nn.Conv1d(6, 64, kernel_size=1),
                                      nn.GELU(),
                                      nn.Conv1d(64, 128, kernel_size=1))
        self.g_feat_mlp = nn.Sequential(nn.Conv1d(self.code_dim, 256, kernel_size=1),  # this will also need adjusting
                                      nn.GELU(),
                                      nn.Conv1d(256, 128, kernel_size=1))
        self.trsf1 = TransformerBlock(c_in=256, c_out=256, num_heads=4, ff_dim=256*2)
        self.trsf2 = TransformerBlock(c_in=256, c_out=256, num_heads=4, ff_dim=256*2)
        self.trsf3 = TransformerBlock(c_in=256, c_out=256*self.scale, num_heads=4, ff_dim=256*2) # this ff_dim shd b x4 0r x8

        # if rf_level == 2, comment this trsf blk
        self.trsf4 = TransformerBlock(c_in=256*self.scale, c_out=128*self.scale, num_heads=4, ff_dim=128*4)
        self.conv1 = nn.Conv1d(128*self.scale, 128*self.scale, kernel_size=1)

        # self.conv1 = nn.Conv1d(256*self.scale, 256*self.scale, kernel_size=1)
        self.last_mlp = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1),  # +128
                                      nn.Conv1d(128, 64, kernel_size=1),
                                      nn.GELU(),
                                      nn.Conv1d(64, 6, kernel_size=1))

        if self.rf_level == 2:
            scale2 = 2
            self.seed_mlp1 = nn.Sequential(nn.Conv1d(6, 64, kernel_size=1),
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
                                         nn.Conv1d(64, 6, kernel_size=1))


    def forward(self, partial_pc, glob_feat):  # glob_feat shd be 1024, thus seed of 1024

        coarse = self.mlp(glob_feat).reshape(-1, self.num_coarse, 6)      # [B, num_coarse, 6] coarse point cloud
        ctrl_x = torch.cat([partial_pc.permute(0, 2, 1), coarse], dim=1)  # [B, N+num_coarse, 6]
        fps_idx = pointnet2_utils.furthest_point_sample(ctrl_x[:, :, :3].contiguous(), 1024) 
        ctrl_x = ctrl_x.permute(0,2,1).contiguous()
        seed = pointnet2_utils.gather_operation(ctrl_x, fps_idx.int())    # [B, 6, N]
        del ctrl_x
        torch.cuda.empty_cache()

        #since we have to concat the shape code & seed along channel dim, need to make sure dims balance/match
        seed = self.seed_mlp(seed)
        #TODO: PointConv Layer can go here, that is if rf_level=1, else it shd b btwn the two refine steps
        glob_feat1 = self.g_feat_mlp(glob_feat.unsqueeze(-1))
        seed_gf = torch.cat([seed, glob_feat1.repeat(1, 1, seed.size(2))], dim=1)

        seed_gf = self.trsf1(seed_gf)  # [B, c_out, N]
        seed_gf = self.trsf2(seed_gf)  # [B, c_out, N]
        seed_gf = self.trsf3(seed_gf)  # [B, c_out*scale, N]

        seed_gf = self.trsf4(seed_gf)  # [B, c_out/2 * scale, N]

        B, N, _ = coarse.shape
        seed_gf = self.conv1(seed_gf).reshape(B, -1, N*self.scale)  # [B, c_out/2, N*scale]

        #bcoz seed_gf is potentially upped by self.scale, seed must also be upped to match bf concat & mlp
        seed_gf = torch.cat([seed_gf, seed.repeat(1, 1, self.scale)], dim=1)
        seed_gf_m = self.last_mlp(seed_gf)

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
            #TODO: extra trsf layer
            return coarse, seed_gf_m, seed_gf

class PCCNet(nn.Module):
    def __init__(self, kmax=20, code_dim=512):
        super().__init__()
        self.k = kmax
        self.code_dim = code_dim
        self.enc = PCEncoder(kmax=self.k, code_dim=self.code_dim)
        self.dec = PCDecoder(code_dim=code_dim, scale=4, rf_level=1)

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
        cdt_coarse, cdp_coarse, cdt_fine, cdp_fine, cdt_finer, cdp_finer = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for i, data in enumerate(loader):
            #data
            xyz = data[0][:, :, :6].to(device).float()  # partial: [B 2048, 6] include normals
            #model
            coarse, fine, finer = model(xyz)
            coarse, fine = coarse[:, :, :3], fine[:, :, :3]
            #losses
            gt_xyz = data[1][:, :, :3].to(device).float()  # partial: [B 16348, 6]
            if finer is not None:
                finer = finer[:, :, :3]
                cdp_finer += chamfer_loss_sqrt(finer, gt_xyz).item()  #inputs shd be BNC; cd_p
                cdt_finer += chamfer_loss(finer, gt_xyz).item() 
            else:
                cdp_finer, cdt_finer = 0.0, 0.0
            cdp_fine += chamfer_loss_sqrt(fine, gt_xyz).item()  #inputs shd be BNC; cd_p
            cdp_coarse += chamfer_loss_sqrt(coarse, gt_xyz).item()  
            cdt_fine += chamfer_loss(fine, gt_xyz).item()  # cd_t
            cdt_coarse += chamfer_loss(coarse, gt_xyz).item()  

            if rand_save and args.max_epoch == epoch and i==0:
                if finer is not None:
                    np.savez(str(args.file_dir) + '/rand_outs.npz', gt_pnts=gt_xyz.data.cpu().numpy(), 
                                                                    final_pnts=finer.data.cpu().numpy(), 
                                                                    fine_pnts=fine.data.cpu().numpy(), 
                                                                    coarse_pnts=coarse.data.cpu().numpy(),
                                                                    als_pnts=xyz.data.cpu().numpy()[:, :, :3])
                else:
                    np.savez(str(args.file_dir) + '/rand_outs.npz', gt_pnts=gt_xyz.data.cpu().numpy(),
                                                                    final_pnts=fine.data.cpu().numpy(), 
                                                                    coarse_pnts=coarse.data.cpu().numpy(),
                                                                    als_pnts=xyz.data.cpu().numpy()[:, :, :3])

    return {'finer_p': cdp_finer/num_iters, 'fine_p': cdp_fine/num_iters, 'coarse_p': cdp_coarse/num_iters, 'finer_t': cdt_finer/num_iters, 'fine_t': cdt_fine/num_iters, 'coarse_t': cdt_coarse/num_iters}
