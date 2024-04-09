import torch, os, sys
import numpy as np
import trimesh
from mesh_to_sdf import get_surface_point_cloud
from point_ops.pointnet2_ops import pointnet2_utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = '/data/processed/2048'
save_path = '/data/under70'
mesh_path = '/data/processed/2048/03_nnt_obj'
partial_path = '/data/processed/2048/fixed_als_txt'

mesh_flist = os.listdir(mesh_path)
partial_flist = os.listdir(partial_path)

omesh_list = [i for i in mesh_flist if i.startswith('original')]
opartial_list = [i for i in partial_flist if i.startswith('original')]

for i in range(len(omesh_list)):
    mesh_file = omesh_list[i]
    if mesh_file.replace('.obj', '.txt.npz') in opartial_list:
        partial_file = mesh_file.replace('.obj', '.txt.npz')
    else:
        print('partial file not found for: ', mesh_file)
        continue
    print(partial_file, " | ", mesh_file)

    partial_pc = np.load(os.path.join(partial_path, partial_file))['unit_als']  # has normals
    np.savez(os.path.join(save_path, 'xyz', mesh_file[:-4] + '.npz'), unit_als=partial_pc)

    mesh = trimesh.load(os.path.join(mesh_path, mesh_file))

    # complete_pc = trimesh.sample
    '''surf_instance contains various data, e.g. points, kd_tree, etc.'''
    surf_instance = get_surface_point_cloud(mesh, 
                                            surface_point_method='sample', 
                                            bounding_radius=1, 
                                            scan_count=30, 
                                            scan_resolution=200, 
                                            sample_point_count=50000, 
                                            calculate_normals=True)

    # use fps to reduce points to fixed number
    surf_pnt_samples = torch.from_numpy(surf_instance.points).float()[None, :, :].to(device) # BN3
    surf_pnt_normals = torch.from_numpy(surf_instance.normals).float()[None, :, :].to(device) # BN3
    fps_idx = pointnet2_utils.furthest_point_sample(surf_pnt_samples, 16348) # xyz: torch.Tensor

    surf_pnt_samples = surf_pnt_samples.permute(0,2,1).contiguous()
    complete_pc = pointnet2_utils.gather_operation(surf_pnt_samples, fps_idx.int())
    complete_normals = pointnet2_utils.gather_operation(surf_pnt_normals.permute(0,2,1).contiguous(), fps_idx.int())

    complete_pc = torch.cat((torch.squeeze(complete_pc), torch.squeeze(complete_normals)),0).permute(1,0) # add .copy() to this line and change variable name if previous line has more use down the line
    
    # save complete_pc  as .txt
    np.savetxt(os.path.join(save_path, 'gt_xyz', mesh_file[:-4] + '.txt'), complete_pc.cpu().numpy(), fmt='%.6f %.6f %.6f %.6f %.6f %.6f')

    # os copy partial_pc file from partial_path to save_path
    # os.system('cp ' + os.path.join(partial_path, partial_file) + ' ' + os.path.join(save_path, 'xyz', partial_file))

# print counts
print(len(os.listdir(os.path.join(save_path, 'gt_xyz'))))
print(len(os.listdir(os.path.join(save_path, 'xyz'))))
    

