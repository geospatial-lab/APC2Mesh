import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, random
from dataset_pcc import CustomDataset
from torch.utils import data
from network import PCCNet, validate
from point_ops.pointnet2_ops import pointnet2_utils as p2u
# from torch.utils.tensorboard import SummaryWriter
import pytorch_warmup as warmup
from loss_pcc import chamfer_loss_sqrt, l2_normal_loss, density_cd
from config_pcc import Args as args
from config_pcc import start_logger
from pytictoc import TicToc


seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tr_dataset = CustomDataset(split='ablation_tr', npoints=args.npoints, device=device)
tr_loader = data.DataLoader(tr_dataset, batch_size=args.bs, shuffle=True)

ts_dataset = CustomDataset(split='ablation_ts', npoints=args.npoints, device=device)
ts_loader = data.DataLoader(ts_dataset, batch_size=args.bs, shuffle=False)

pcc_model = PCCNet(kmax=20, code_dim=1024, use_nmls=True, 
                   multi_scale=False, attn_pool=True, fps_crsovr=True).to(device)

if args.load_chkpnt:
    optimizer = torch.optim.AdamW(pcc_model.parameters(), lr=args.fix_lr, betas=(0.9, 0.999),
                                weight_decay=args.wd)  #eps=1e-08,  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.93, patience=1, verbose=True)
else:
    optimizer = torch.optim.AdamW(pcc_model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                weight_decay=args.wd)  #eps=1e-08, 

    num_steps = len(tr_loader) * args.max_epoch
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=args.eta_min)  #TODO: try steplr &eponentiallr too
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

# loss init
t = TicToc() #create instance of class
llr_logger = start_logger(log_dir=args.log_dir, fname='lr_loss')
if args.load_chkpnt:
    pcc_model.load_state_dict(torch.load(args.chkpnt_path))
    llr_logger.info('model loaded and training continues from: %s'%(args.chkpnt_path))
llr_logger.info('ms:[10,20,30] | 256-trsf | 4heads | scale: x4 | bs: 8 | #ep: 140 |lr: 0.0006 | eta_min: 1e-07 | normals: Yes | #tr/ts: 2k/40 | tr_loss: %s' %(args.tr_loss))
llr_logger.info('fps_crsovr: %s | attn_pool: %s | multiscale: %s | use_nmls: %s ' %(True, True, False, True))
# tb = SummaryWriter(comment=f'ms:[10,20,30] | 256-trsf | 4heads | scale: [x4, +ppconv]')

init_epoch = int(args.chkpnt_path[-23:-20]) if args.load_chkpnt else 1
max_epoch = init_epoch + 60 if args.load_chkpnt else args.max_epoch
best_val_cdp = 20.0
best_val_cdt = 20.0
coarse_list, fine_list, total_list = [], [], []
for epoch in range(init_epoch, max_epoch+1):
    avg_time_per_iter = []
    for i, data in enumerate(tr_loader, 0):
        t.tic() #Start timer
        optimizer.zero_grad()
        xyz = data[0][:, :, :6].to(device).float()  # partial: [B 2048, 6] include normals
        if not pcc_model.use_nmls:
            xyz = xyz[:, :, :3]
        pcc_model.train()
        coarse, fine, finer = pcc_model(xyz)

        # Loss (xyz)
        gt_xyz = data[1][:, :, :6].to(device).float()  # partial: [B 16348, 6] include normals

        gt_fine = p2u.gather_operation(gt_xyz.permute(0,2,1).contiguous(), p2u.furthest_point_sample(gt_xyz[:,:,:3].contiguous(), fine.size(1))).permute(0,2,1)
        if args.tr_loss == 'dcd':
            loss_fine = density_cd(fine[:, :, :3], gt_fine[:, :, :3], alpha=args.t_alpha, n_lambda=args.n_lambda)
        elif args.tr_loss == 'cd':
            loss_fine = chamfer_loss_sqrt(fine[:, :, :3], gt_fine[:, :, :3]) * 10  #inputs shd be BNC

        gt_coarse = p2u.gather_operation(gt_fine.permute(0,2,1).contiguous(), p2u.furthest_point_sample(gt_fine[:,:,:3].contiguous(), coarse.size(1))).permute(0,2,1)
        if args.tr_loss == 'dcd':
            loss_coarse = density_cd(coarse[:, :, :3], gt_coarse[:, :, :3], alpha=args.t_alpha, n_lambda=args.n_lambda)
        elif args.tr_loss == 'cd':
            loss_coarse = chamfer_loss_sqrt(coarse[:, :, :3], gt_coarse[:, :, :3]) * 10
        
        loss = loss_coarse + loss_fine
        if pcc_model.use_nmls:
            normal_loss = l2_normal_loss(gt_fine, fine)
        else:
            normal_loss = 0.0

        coarse_list.append(loss_coarse)
        fine_list.append(loss_fine)

        # Loss (normals)
        # nbr_pnt_loss, nbr_norm_loss = nbrhood_uniformity_loss(fine, 10, 10)
        loss = loss + normal_loss #0.1* # + nbr_pnt_loss + nbr_norm_loss

        # if args.p2p and (epoch % args.p2p_interval) == 0:
        #     loss += p2p_dist(gt_fine, fine) * 0.1

        total_list.append(loss)

        if i % args.record_step == 0:
            
            iter_time = 0.0 if i == 0 else sum(avg_time_per_iter)/len(avg_time_per_iter)
            llr_logger.info('Epoch %.3d | iter %.3d/%d, %.5f secs | l_coarse: %.6f | l_fine: %.6f | l_total: %.6f | lrs: %.10f | c_lr: %.10f' % (epoch, i, 
                                                                                            len(tr_dataset)/args.bs, 
                                                                                            iter_time,
                                                                                            (sum(coarse_list)/len(coarse_list)).item(),
                                                                                            (sum(fine_list)/len(fine_list)).item(), 
                                                                                            (sum(total_list)/len(total_list)).item(),
                                                                                            0.0 if args.load_chkpnt else warmup_scheduler.lrs[0], 
                                                                                            optimizer.param_groups[0]['lr']))
            coarse_list, fine_list, total_list = [], [], []
            #TODO: push record_step info it to tb for graphing or retrieve n plot from log files

        loss.backward()
        optimizer.step()
        
        if not args.load_chkpnt:
            with warmup_scheduler.dampening():
                lr_scheduler.step()

        avg_time_per_iter.append(t.tocvalue()) # t.tocvalue: time elapsed since t.tic()

    if (epoch % args.val_interval == 0) or (epoch == args.max_epoch):  # bcoz max_epoch above is +1
        val_losses = validate(pcc_model, ts_loader, epoch, args, device=loss.device, rand_save=True) 
        '''first two values after vEpoch will be zero if args.tr_loss == 'cd' '''
        llr_logger.info('vEpoch %.3d | %s_fine: %.6f | %s_coarse: %.6f |cdp_fine: %.6f | cdt_fine: %.6f | cdp_coarse: %.6f | cdt_coarse: %.6f' %(epoch,
                                                                                                        args.tr_loss,
                                                                                                        val_losses['fine_d'], 
                                                                                                        args.tr_loss,
                                                                                                        val_losses['coarse_d'],
                                                                                                        val_losses['fine_p'], 
                                                                                                        val_losses['fine_t'],
                                                                                                        val_losses['coarse_p'],
                                                                                                        val_losses['coarse_t']))
        if args.load_chkpnt:
            scheduler.step(val_losses['fine_p'])
        # if args.tr_loss == 'dcd' and (val_losses['fine_d'] < best_val_cdp) or (val_losses['coarse_d'] < best_val_cdt):
        #     best_val_cdp = val_losses['fine_d']
        #     best_val_cdt = val_losses['coarse_d'] 
        #     torch.save(pcc_model.state_dict(), '%s/pccnet_%.3d_%.5f_%.5f.pth' % (str(args.ckpts_dir), epoch, best_val_cdp, best_val_cdt))
       
        if (val_losses['fine_p'] < best_val_cdp) or (val_losses['fine_t'] < best_val_cdt):
            best_val_cdp = val_losses['fine_p']
            best_val_cdt = val_losses['fine_t']
            torch.save(pcc_model.state_dict(), '%s/pccnet_%.3d_%.5f_%.5f.pth' % (str(args.ckpts_dir), epoch, best_val_cdp, best_val_cdt))
        print("Saving model...")

print('done ...')

#TODO: (1) w/o normals: input, output, loss (2) w/o multi-scale (3) w/o FPS cross-over

