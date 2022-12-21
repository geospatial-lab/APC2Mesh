import torch, random
from dataset_pcc import CustomDataset
from torch.utils import data
from network import PCCNet, validate
from point_ops.pointnet2_ops import pointnet2_utils as p2u
from torch.utils.tensorboard import SummaryWriter
import pytorch_warmup as warmup
from loss_pcc import chamfer_loss_sqrt
from config_pcc import Args as args
from config_pcc import start_logger
from pytictoc import TicToc


seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tr_dataset = CustomDataset(split='train', npoints=args.npoints, device=device)
tr_loader = data.DataLoader(tr_dataset, batch_size=args.bs, shuffle=True)

ts_dataset = CustomDataset(split='test', npoints=args.npoints, device=device)
ts_loader = data.DataLoader(ts_dataset, batch_size=args.bs, shuffle=False)

pcc_model = PCCNet(kmax=20, code_dim=512).to(device)

optimizer = torch.optim.AdamW(pcc_model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                             weight_decay=args.wd)  #eps=1e-08, 

num_steps = len(tr_loader) * args.max_epoch
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)  #TODO: try steplr &eponentiallr too
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
# self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
# self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
#                                         after_scheduler=self.scheduler_steplr)

# loss init
t = TicToc() #create instance of class
llr_logger = start_logger(log_dir=args.log_dir, fname='lr_loss')
llr_logger.info('ms:[10,20,30] | 256-trsf | 4heads | scale: [x4,] | bs: 8 | #ep: 80 | normals: No | #tr/ts: 2000/40')
tb = SummaryWriter(comment=f'ms:[10,20,30] | 256-trsf | 4heads | scale: [x4,]')

init_epoch = 1  # for now. #TODO: adapt it to pick it from pretrain output filename later
best_val_cdp = 1.0
best_val_cdt = 1.0
coarse_list, fine_list, total_list = [], [], []
for epoch in range(init_epoch, args.max_epoch+1):
    avg_time_per_iter = []
    for i, data in enumerate(tr_loader, 0):
        t.tic() #Start timer
        optimizer.zero_grad()
        xyz = data[0][:, :, :3].to(device).float()  # partial: [B 2048, 6] include normals

        pcc_model.train()
        coarse, fine = pcc_model(xyz)

        # Loss
        gt_xyz = data[1][:, :, :3].to(device).float().contiguous()  # partial: [B 16348, 6] include normals
        gt_fine = p2u.gather_operation(gt_xyz.permute(0,2,1).contiguous(), p2u.furthest_point_sample(gt_xyz, fine.size(1))).permute(0,2,1)
        loss_fine = chamfer_loss_sqrt(fine, gt_fine)  #inputs shd be BNC

        gt_coarse = p2u.gather_operation(gt_fine.permute(0,2,1).contiguous(), p2u.furthest_point_sample(gt_fine.contiguous(), coarse.size(1))).permute(0,2,1)
        loss_coarse = chamfer_loss_sqrt(coarse, gt_coarse)

        loss = (loss_coarse + loss_fine)/2

        coarse_list.append(loss_coarse)
        fine_list.append(loss_fine)
        total_list.append(loss)
        #TODO: Run a normal loss function on final predicted point against mesh-sampled pc normals. i.e., final outcome = [x y z nx ny nz]
        #TODO: or add normals to input to make it BN6 right from the beginning and learning the normals base on a loss func.
        if i % args.record_step == 0:
            
            iter_time = 0.0 if i == 0 else sum(avg_time_per_iter)/len(avg_time_per_iter)
            llr_logger.info('Epoch %.3d | iter %.3d/%d, %.5f secs | l_coarse: %.6f | l_fine: %.6f | l_total: %.6f | lrs: %.10f | c_lr: %.10f' % (epoch, i, 
                                                                                            len(tr_dataset)/args.bs, 
                                                                                            iter_time,
                                                                                            (sum(coarse_list)/len(coarse_list)).item(),
                                                                                            (sum(fine_list)/len(fine_list)).item(), 
                                                                                            (sum(total_list)/len(total_list)).item(),
                                                                                            warmup_scheduler.lrs[0], 
                                                                                            optimizer.param_groups[0]['lr']))
            coarse_list, fine_list, total_list = [], [], []
            #TODO: push record_step info it to tb for graphing or retrieve n plot from log files

        loss.backward()
        optimizer.step()
        
        with warmup_scheduler.dampening():
            lr_scheduler.step()

        avg_time_per_iter.append(t.tocvalue()) # t.tocvalue: time elapsed since t.tic()

    if (epoch % args.val_interval == 0) or (epoch == args.max_epoch):  # bcoz max_epoch above is +1
        val_losses = validate(pcc_model, ts_loader, epoch, args, device=loss.device, rand_save=True) 

        llr_logger.info('vEpoch %.3d | cdp_fine: %.6f | cdt_fine: %.6f | cdp_coarse: %.6f | cdt_coarse: %.6f' %(epoch, 
                                                                                                       val_losses['fine_p'], 
                                                                                                       val_losses['fine_t'],
                                                                                                       val_losses['coarse_p'],
                                                                                                       val_losses['coarse_t']))

        if (val_losses['fine_p'] < best_val_cdp) or (val_losses['fine_t'] < best_val_cdt):
            best_val_cdp = val_losses['fine_p']
            best_val_cdt = val_losses['fine_t']
            torch.save(pcc_model.state_dict(), '%s/pccnet_%.3d_%.5f_%.5f.pth' % (str(args.ckpts_dir), epoch, best_val_cdp, best_val_cdt))
            print("Saving model...")

print('done ...')

#TODO: ideas
'''
(*) include normal data n loss (maybe this will boost edge/corner awareness)
(*) add a second refine step
(1) increase training data
(2) increase training epochs
(3) currently, c_lr/lrs goes all the way to 0, find a way to stick to a fix mininum 
'''

