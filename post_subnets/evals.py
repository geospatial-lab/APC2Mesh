from post_ops import *
from glob import glob

trsf_path = "/data/processed/2048/03_trsf_npz"
rt_dir = "/outputs/experiments/testing/2023-03-29_23-26"
save_dir = f"{rt_dir}/complete_txt"

pcc_blist = glob(f'{rt_dir}/*.npz')

#retrieve individual completed point files from the batched .npz files 
if not os.path.exists(f'{save_dir}/als') or not os.listdir(f'{save_dir}/als'):
    get_complete_files(pcc_blist, rt_dir, save_dir)

# since point/meshes were normalized to [-1, +1] before learning, denormalize to original geo-coords
denormalize(save_dir, trsf_path)

# since pcc test computed metrics for whole tests, now cumpute fron individual instances
get_pcc_errors()

print('done !')