import torch, random
from dataset_pcc import CustomDataset
from torch.utils import data
from network import PCCNet
# from point_ops.pointnet2_ops import pointnet2_utils as p2u
from loss_pcc import chamfer_loss_sqrt, chamfer_loss
from pytictoc import TicToc
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Params - files
chkpnt_path = '/outputs/experiments/2023-03-21_08-01/checkpoints/pccnet_112_0.01676_0.00083.pth'

experiment_dir = Path('/outputs/experiments/testing/')
experiment_dir.mkdir(exist_ok=True)
file_dir = Path(str(experiment_dir) + '/' + str(datetime.now().strftime('%Y-%m-%d_%H-%M')))
file_dir.mkdir(exist_ok=True)
log_dir = file_dir.joinpath('logs/')
log_dir.mkdir(exist_ok=True) 

def start_logger(log_dir, fname):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # logging to file
    file_handler = logging.FileHandler(str(log_dir) + '/%s.txt'%(fname))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))  # %(asctime)s - %(levelname)s -
    logger.addHandler(file_handler)

    # logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('\t\t %(message)s'))
    logger.addHandler(stream_handler)

    return logger

# Params - others
bs = 8  # batch_size
npoints = 2048

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# tr_dataset = CustomDataset(split='train', npoints=npoints, device=device)
# tr_loader = data.DataLoader(tr_dataset, batch_size=bs, shuffle=True)

ts_dataset = CustomDataset(split='test', npoints=npoints, device=device)
ts_loader = data.DataLoader(ts_dataset, batch_size=bs, shuffle=False)

pcc_model = PCCNet(kmax=20, code_dim=1024).to(device)

pcc_model.load_state_dict(torch.load(chkpnt_path))
print('loaded model: %s' % chkpnt_path)

t = TicToc() #create instance of class
test_logger = start_logger(log_dir=log_dir, fname='test_log')

def testing(model, loader, file_dir, device, rand_save=False): 
    print("Testing ...")
    model.eval()
    num_iters = len(loader)

    with torch.no_grad():
        cdt_coarse, cdp_coarse, cdt_fine, cdp_fine, cdt_finer, cdp_finer = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for i, data in enumerate(loader):
            print(i,'\n')
            continue
            #data
            xyz = data[0][:, :, :6].to(device).float()  # partial: [B 2048, 6] include normals

            #model
            coarse, fine, finer = model(xyz)
            # coarse, fine = coarse[:, :, :3], fine[:, :, :3]

            #losses
            gt_xyz = data[1][:, :, :3].to(device).float()  # partial: [B 16348, 6]
            if finer is not None:
                # finer = finer[:, :, :3]
                cdp_finer += chamfer_loss_sqrt(finer[:, :, :3], gt_xyz).item()  #inputs shd be BNC; cd_p
                cdt_finer += chamfer_loss(finer[:, :, :3], gt_xyz).item() 
            else:
                cdp_finer, cdt_finer = 0.0, 0.0
            cdp_fine += chamfer_loss_sqrt(fine[:, :, :3], gt_xyz).item()  #inputs shd be BNC; cd_p
            cdp_coarse += chamfer_loss_sqrt(coarse[:, :, :3], gt_xyz).item()  
            cdt_fine += chamfer_loss(fine[:, :, :3], gt_xyz).item()  # cd_t
            cdt_coarse += chamfer_loss(coarse[:, :, :3], gt_xyz).item()  

            if rand_save:
                # if finer is not None:
                #     np.savez(str(file_dir) + '/rand_outs.npz', gt_pnts=gt_xyz.data.cpu().numpy(), 
                #                                                     final_pnts=finer.data.cpu().numpy(), 
                #                                                     fine_pnts=fine.data.cpu().numpy(), 
                #                                                     coarse_pnts=coarse.data.cpu().numpy(),
                #                                                     als_pnts=xyz.data.cpu().numpy()[:, :, :3])
                # else:                                                 
                np.savez(str(file_dir) + '/rand_outs_{}.npz'.format(i), gt_pnts=data[1].cpu().numpy(),
                                                                        final_pnts=fine.data.cpu().numpy(), 
                                                                        coarse_pnts=coarse.data.cpu().numpy(),
                                                                        als_pnts=xyz.data.cpu().numpy()[:, :, :3])

    return {'finer_p': cdp_finer/num_iters, 'fine_p': cdp_fine/num_iters, 'coarse_p': cdp_coarse/num_iters, 'finer_t': cdt_finer/num_iters, 'fine_t': cdt_fine/num_iters, 'coarse_t': cdt_coarse/num_iters}


test_losses = testing(pcc_model, ts_loader, file_dir=file_dir, device=device, rand_save=True) 

test_logger.info('cdp_finer: %.6f | cdt_finer: %.6f |cdp_fine: %.6f | cdt_fine: %.6f | cdp_coarse: %.6f | cdt_coarse: %.6f' %(
                                                                            test_losses['finer_p'], 
                                                                            test_losses['finer_t'],
                                                                            test_losses['fine_p'], 
                                                                            test_losses['fine_t'],
                                                                            test_losses['coarse_p'],
                                                                            test_losses['coarse_t']))
# get file name and sequence for future de-normalization step.
ts_fileseq = ts_loader.batch_sampler.sampler.data_source.mesh_list 
# open file in write mode
with open(str(file_dir)+'/ts_fileseq.txt', 'w') as fp:
    for item in ts_fileseq:
        # write each item on a new line
        fp.write("%s\n" % item)
    # print('Done')
print('done ...')

