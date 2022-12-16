import torch, random
from dataset_pcc import CustomDataset
from torch.utils import data
from network import PCCNet
from point_ops.pointnet2_ops import pointnet2_utils as p2u
from torch.utils.tensorboard import SummaryWriter
from loss_pcc import chamfer_loss_sqrt

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 8  #TODO: move later to config file
BuildingDataset = CustomDataset(npoints=2048, device=device)
tr_loader = data.DataLoader(BuildingDataset, batch_size=8, shuffle=True)

pcc_model = PCCNet(kmax=20, code_dim=512).to(device)

optimizer = torch.optim.Adam(pcc_model.parameters(), lr=0.0001, betas=(0.9, 0.999),
                             eps=1e-08, weight_decay=1e-06)
#TODO: scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=2, verbose=True)
# lr scheduler
# self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
# self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
#                                         after_scheduler=self.scheduler_steplr)

# loss init

init_epoch = 0  # for now. #TODO: adapt it to pick it from pretrain output filename later
max_epoch = 40
record_step = 10

tb = SummaryWriter(comment=f'ms:[10,20,30] | 256-trsf | 4heads | scale: [4,]')
for epoch in range(init_epoch, max_epoch):
    
    for i, data in enumerate(tr_loader, 0):
        optimizer.zero_grad()
        xyz = data[0][:, :, :3].to(device).float()  # partial: [B 2048, 6] include normals

        pcc_model.train()
        coarse, fine = pcc_model(xyz)

        gt_xyz = data[1][:, :, :3].to(device).float().contiguous()  # partial: [B 16348, 6] include normals
        gt_fine = p2u.gather_operation(gt_xyz.permute(0,2,1).contiguous(), p2u.furthest_point_sample(gt_xyz, fine.size(2))).permute(0,2,1)
        loss_fine = chamfer_loss_sqrt(fine.permute(0,2,1), gt_fine)  #inputs shd be BNC

        gt_coarse = p2u.gather_operation(gt_fine.permute(0,2,1).contiguous(), p2u.furthest_point_sample(gt_fine, fine.size(2)))
        loss_coarse = chamfer_loss_sqrt(coarse, gt_coarse)

        loss = torch.mean([loss_coarse, loss_fine])

        #TODO: Run a normal loss function on final predicted point against mesh-sampled pc normals. i.e., final outcome = [x y z nx ny nz]
        #TODO: or add normals to input to make it BN6 right from the beginning and learning the normals base on a loss func.
        if i%record_step == 0:
            print('\nEpoch %d | iter %d / %d | loss_coarse: %f | loss_fine: %f | loss_total: %f' % (epoch, i, 
                                                                                                    len(BuildingDataset)/batch_size, 
                                                                                                    loss_coarse.item(),
                                                                                                    loss_fine.item(), 
                                                                                                    loss.item()))

            #TODO: create a list above & avg every record_step & push it to tb for graphing

        loss.backward()
        optimizer.step()

        print('done ...')

#val goes here