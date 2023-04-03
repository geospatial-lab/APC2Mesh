'''classes and functions for pcc and p2m net evaluations'''
import os, sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh

def get_list_per_batch(bs, bseq):
    for i in range(0, len(bseq), bs):
        yield bseq[i : i+bs]

def get_complete_files(blist, rt_dir, save_dir):
    try:
        with open(f"{rt_dir}/ts_fileseq.txt", 'r') as f:
            bfiles_seq = f.readlines()
    except FileNotFoundError:
        print(f'File {rt_dir}/ts_fileseq.txt not found!', file=sys.stderr)
        return

    if not os.path.exists(save_dir):
        os.makedirs(f"{save_dir}/gt") 
        os.makedirs(f"{save_dir}/fine")
        os.makedirs(f"{save_dir}/als")

    list_per_batch = list(get_list_per_batch(bs=8, bseq=bfiles_seq))

    for bname in blist:
        pcc_out = np.load(bname)  # 8 files per batch
        fine = pcc_out['final_pnts']
        gt = pcc_out['gt_pnts']
        als = pcc_out['als_pnts']

        del pcc_out
        b_num = int(os.path.splitext(os.path.basename(bname))[0].split('_')[-1])

        for i in range(len(fine)):  #which is equal to batch size of 8
            temp_c = np.squeeze(gt[i, :, :])
            np.savetxt(f"{save_dir}/gt/" + list_per_batch[b_num][i].strip()[:-4] + '.txt', temp_c)
            temp_f = np.squeeze(fine[i, :, :])
            np.savetxt(f"{save_dir}/fine/" + list_per_batch[b_num][i].strip()[:-4] + '.txt', temp_f)
            temp_als = np.squeeze(als[i, :, :])
            np.savetxt(f"{save_dir}/als/" + list_per_batch[b_num][i].strip()[:-4] + '.txt', temp_als)


def normalize_mesh(file_in, file_out=None):

        mesh = trimesh.load(file_in)

        # Calculate centroid
        centroid = mesh.vertices.mean(0)

        # translate to origin
        recenteredData = mesh.vertices - centroid

        # scale to unit sphere
        scale = np.abs(recenteredData).max() # multiply by 2 if u want range [-0.5, 0.5]
        normalized_data = np.divide(recenteredData, scale)
        outmesh = trimesh.Trimesh(vertices=normalized_data, faces=mesh.faces)
        outmesh.export(file_out)

def normalize_txt(data):
        
        # Calculate centroid
        centroid = data[:, :3].mean(0)

        # translate to origin
        recenteredData = data[:, :3] - centroid

        # scale to unit sphere
        scale = np.abs(recenteredData).max() # multiply by 2 if u want range [-0.5, 0.5]
        normalized_data = np.divide(recenteredData, scale)

        return normalized_data


def denormalize(data_dir, trsf_path):
   
    # als_list = os.listdir(f'{data_dir}/als')
    fine_list = os.listdir(f'{data_dir}/fine')
    trsf_list = os.listdir(trsf_path) 

    for file in fine_list:
        als_abs = f'{data_dir}/als/{file}'
        fine_abs = f'{data_dir}/fine/{file}'
        if file[:-4]+'.npz' in trsf_list:
            trsf_abs = trsf_path + '/' + file[:-4] + '.npz' 

        als_pnts = np.loadtxt(als_abs)
        fine_pnts = np.loadtxt(fine_abs)
        trsf_data = np.load(trsf_abs)

        #re-normalize fine_pnts, bcoz some completed points could be out of the original [-1,+1] bounds
        re_norm_xyz = normalize_txt(fine_pnts)

        # un-normalize the data
        rescaled_als = np.multiply(als_pnts, trsf_data['scale'])
        rescaled_finexyz = np.multiply(re_norm_xyz, trsf_data['scale'])  

        translated_als = rescaled_als + trsf_data['centroid']
        translated_fine = rescaled_finexyz + trsf_data['centroid']
        fine_pnts[:,:3] = translated_fine

        #TODO: check correctness of denormalized data by comparing to the original
        # print(np.allclose(de_normdata,orig_data))

        #create dir and save files
        denorm_dir = os.path.join(os.path.split(data_dir)[0], 'denorm_txt') 
        if not os.path.exists(denorm_dir): 
            os.makedirs(f"{denorm_dir}/fine")
            os.makedirs(f"{denorm_dir}/als")

        np.savetxt(f"{denorm_dir}/fine/{file}", fine_pnts, fmt='%.4f %.4f %.4f %.6f %.6f %.6f')
        np.savetxt(f"{denorm_dir}/als/{file}", translated_als, fmt='%.4f %.4f %.4f')

def get_pcc_errors():
    pass
         
def get_dist_losses(rec_list, gt_list):

    for (rec_file, gt_file) in zip(rec_list, gt_list):

        gt = np.loadtxt(gt_file)
        # Pass gt's xyz to Open3D.o3d.geometry.PointCloud
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt)
        # o3d.visualization.draw_geometries([gt_pcd])

        # load and render mesh
        mesh = o3d.io.read_triangle_mesh(rec_file)
        # o3d.visualization.draw_geometries([mesh])
        rec_pcd = mesh.sample_points_uniformly(number_of_points=16348)
        # o3d.visualization.draw_geometries([rec_pcd])

        tree = cKDTree(gt)
        dist, idx = tree.query(np.asarray(rec_pcd.points), k=1)
        euc_dist = np.sum(np.square(np.asarray(rec_pcd.points) - gt[idx,:]), axis=1)
        # metric_out = np.sum(np.square(nbrs[:, :, 3:] - np.expand_dims(query_pnts, axis=1)[:, :, 3:]), axis=-1)
        euc_dist = np.sqrt(euc_dist).reshape(-1,1)

        #TODO: Add the error for normals

        # dist_err_pcd = o3d.geometry.PointCloud()
        # dist_err_pcd.points = o3d.utility.Vector3dVector(gt)
        # o3d.visualization.draw_geometries([dist_err_pcd])

        xyz_derr = np.column_stack([np.asarray(rec_pcd.points),euc_dist]) #derr: dist_err
        fname = rec_file[:-4]+'.txt'
        np.savetxt(fname, xyz_derr, fmt='%.6f %.6f %.6f %.6f')

print('Done!!')