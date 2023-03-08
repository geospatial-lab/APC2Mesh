import os, logging
import numpy as np
import torch

MANIFOLD_DIR = '/Manifold/build'  # path to manifold software (https://github.com/hjwdzh/Manifold)

class Args(object):

    # HParams - files
    fix_sample_cnt = 2048  # for now [2048, 4096] from sdf_try.py
    data_path = "/data/processed/%s/net_outputs/pcc_out" %(fix_sample_cnt)
    # data_path = '/data/processed/%d' %(fix_sample_cnt)
    # complete_path = os.path.join(data_path, '04_query_npz')
    # partial_path = os.path.join(data_path, '05_als_npz')
    save_path = "/data/processed/%s/net_outputs/p2m_chkpnts" %(fix_sample_cnt)
    os.makedirs(save_path, exist_ok=True)

    # HParams - Rec
    torch_seed = 5
    samples = 5000  # number of points to sample reconstruction with ???
    initial_mesh = None  # if available, replace this with path
    initial_num_faces = 3000
    init_samples = 4000
    iterations = 1000
    upsamp = 1000  # upsample each {upsamp}th iteration
    max_faces = 8000  # maximum number of faces to upsample to
    faces_to_part = [8000, 16000, 20000]  # after how many faces to split

    # HParams - net
    gpu = 0
    lr = 1.1e-4
    ang_wt = 1e-1  # weight of the cosine loss for normals
    res_blks = 3
    lrelu_alpha = 0.01
    local_non_uniform = 0.1  # weight of local non uniform loss
    convs = [16, 32, 64, 64, 128]
    pools = [0.0, 0.0, 0.0, 0.0]  # percent to pool from orig. resolution in each layer')
    transfer_data = True
    overlap = 0  # overlap for bfs
    global_step = False  #perform the optimization step after all the parts are forwarded (only matters if nparts > 2)
    manifold_res = 20000  # resolution for manifold upsampling
    unoriented = False  # take the normals loss term without any preferred orientation
    init_weights = 0.002
    export_interval = 100
    beamgap_iterations = 0  # the num of iters to which the beamgap loss will be calculated
    beamgap_modulo = 1  # skip iterations with beamgap loss, calc beamgap when: iter % (beamgap-modulo) == 0
    manifold_always = True  # always run manifold even when the maximum number of faces is reached

    if not os.path.exists(save_path):
        os.makedirs(save_path)

def get_num_parts(Args, num_faces):
    lookup_num_parts = [1, 2, 4, 8]
    num_parts = lookup_num_parts[np.digitize(num_faces, Args.faces_to_part, right=True)]
    return num_parts

def dtype():
    return torch.float32

def get_num_samples(Args, cur_iter):
    slope = (Args.samples - Args.init_samples) / int(0.8 * Args.upsamp)
    return int(slope * min(cur_iter, 0.8 * Args.upsamp)) + Args.init_samples

#TODO: call this fcn under a listdir, for loop to access test pred per batch
#TODO: allow test to save all batches, not just the first.
def prep_pcc_data(exp_folder, fix_sample_cnt=2048):
    pcc_path = '/outputs/experiments/testing/{}/rand_outs_0.npz'.format(exp_folder)
    p2m_in_dir = "/data/processed/%s/net_outputs/pcc_out" %(fix_sample_cnt)
    os.makedirs(p2m_in_dir, exist_ok=True)
    # folder = os.path.dirname(pcc_path)

    pcc_out = np.load(pcc_path)
    if len(pcc_out.files) > 4:
        coarse = pcc_out['coarse_pnts']
        fine = pcc_out['fine_pnts']
        gt = pcc_out['gt_pnts']
        als = pcc_out['als_pnts']
        finer = pcc_out['final_pnts']
    else:
        coarse = pcc_out['coarse_pnts']
        fine = pcc_out['final_pnts']
        gt = pcc_out['gt_pnts']
        als = pcc_out['als_pnts']
        finer = None
    del pcc_out

    for i in range(len(fine)):
        temp_c = np.squeeze(coarse[i, :, :])
        np.savetxt(p2m_in_dir+'/coarse_{}.txt'.format(i+1), temp_c)
        temp_f = np.squeeze(fine[i, :, :])
        np.savetxt(p2m_in_dir+'/fine_{}.txt'.format(i+1), temp_f)
        temp_gt = np.squeeze(gt[i, :, :])
        np.savetxt(p2m_in_dir+'/gt_{}.txt'.format(i+1), temp_gt)
        temp_als = np.squeeze(als[i, :, :])
        np.savetxt(p2m_in_dir+'/als_{}.txt'.format(i+1), temp_als)
        if finer is not None:
            temp_f1 = np.squeeze(finer[i, :, :])
            np.savetxt(p2m_in_dir+'/finer_{}.txt'.format(i+1), temp_f1)


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
