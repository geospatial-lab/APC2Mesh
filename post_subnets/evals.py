from post_ops import *
from glob import glob

# trsf_path = "/data/processed/2048/03_trsf_npz"
# rt_dir = "/outputs/experiments/testing/2023-04-20_01-33"  #"2023-03-29_23-26
# save_dir = f"{rt_dir}/complete_txt"
# save_dir = "/data/processed/2048/net_outputs/pcc_out/p2p_train"

# pcc_blist = glob(f'{rt_dir}/*.npz')

#retrieve individual completed point files from the batched .npz files 
# if not os.path.exists(f'{save_dir}/als') or not os.listdir(f'{save_dir}/als'):
#     get_complete_files(pcc_blist, rt_dir, save_dir)

# # since point/meshes were normalized to [-1, +1] before learning, denormalize to original geo-coords
# denormalize(save_dir, trsf_path)

data_path = "/data/processed/2048/net_outputs/pcc_out"
save_path = "/data/processed/2048/net_outputs/p2m_rec_obj/config-f3"
rlist = glob(os.path.join(save_path, f'config-f0/rec_*.obj'))
glist = glob(f'{data_path}/gt/*.txt')
# #TODO: ensure the sequence of the two lists are in sync.

# # since pcc test computed metrics for whole tests, now cumpute fron individual instances
# get_dist_losses(rlist, glist)

# compute mesh scene level errors
get_per_scene_errors(save_path)

print('done !')