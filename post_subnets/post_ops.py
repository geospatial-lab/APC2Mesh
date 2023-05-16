'''classes and functions for pcc and p2m net evaluations'''
import os, sys, glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh
from models.layers.mesh import Mesh
from base_utils import mesh_utils

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

    if not os.path.exists(f"{save_dir}/als"):
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
    xyz_errs, nml_errs = [], []
    rec_list_dir = os.path.dirname(rec_list[0])
    for gt_file in gt_list:

        rec_file = 'rec_'+ os.path.basename(gt_file)[:-4] +'.obj'
        rec_file = os.path.join(rec_list_dir, rec_file)
        if rec_file not in rec_list:
            print(f'{rec_file} not found, skipping...')
            continue

        gt = np.loadtxt(gt_file)
        # Pass gt's xyz to Open3D.o3d.geometry.PointCloud
        # gt_pcd = o3d.geometry.PointCloud()
        # gt_pcd.points = o3d.utility.Vector3dVector(gt[:,:3])
        # o3d.visualization.draw_geometries([gt_pcd])

        # load mesh and sample from it
        mesh = Mesh(rec_file, nml=False)
        xyz, normals = mesh_utils.sample_surface(mesh.faces, mesh.vs.unsqueeze(0), 16348)
        xyz = xyz.squeeze(0)
        normals = normals.squeeze(0)
        # convert to numpy array
        xyz = xyz.cpu().numpy()
        normals = normals.cpu().numpy()

        #re-normalize fine_pnts, bcoz some completed points could be out of the original [-1,+1] bounds
        # xyz = normalize_txt(xyz)
        
        tree = cKDTree(gt[:,:3])
        dist, idx = tree.query(np.asarray(xyz), k=1)
        delta_sq = np.square(np.asarray(xyz) - gt[idx,:3])
        euc_dist = np.sum(delta_sq, axis=1) 
        delta_sq = np.square(np.asarray(normals) - gt[idx,3:])
        nml_dist = np.sum(delta_sq, axis=1)
        euc_dist = np.sqrt(euc_dist).reshape(-1,1)
        nml_dist = np.sqrt(nml_dist).reshape(-1,1)

        xyz_errs.append(np.mean(euc_dist))
        nml_errs.append(np.mean(nml_dist))

        # dist_err_pcd = o3d.geometry.PointCloud()
        # dist_err_pcd.points = o3d.utility.Vector3dVector(gt)
        # o3d.visualization.draw_geometries([dist_err_pcd])

        dist_errs = np.column_stack([xyz,euc_dist,nml_dist]) #derr: dist_err
        fname = rec_file[:-4]+'.txt'
        np.savetxt(fname, dist_errs, fmt='%.6f %.6f %.6f %.6f %.6f')
    
    out = ['distancex error: {}'.format(np.mean(xyz_errs)), 'normal error: {}'.format(np.mean(nml_errs))]
    fpath = os.path.dirname(rec_list[0])
    with open(f'{fpath}/error_summary.txt', 'w') as f:
        f.write('\n'.join(out))

def get_per_instance_errors(gt_path, pcc_name, mesh_path):
    gt_file = os.path.join(gt_path, f'{pcc_name}.txt')
    gt = np.loadtxt(gt_file)

    # load mesh and sample from it
    mesh = Mesh(mesh_path, nml=False)
    xyz, normals = mesh_utils.sample_surface(mesh.faces, mesh.vs.unsqueeze(0), 16348)
    xyz = xyz.squeeze(0)
    normals = normals.squeeze(0)
    # convert to numpy array
    xyz = xyz.cpu().numpy()
    normals = normals.cpu().numpy()

    tree = cKDTree(gt[:,:3])
    dist, idx = tree.query(np.asarray(xyz), k=1)
    delta_sq = np.square(np.asarray(xyz) - gt[idx,:3])
    euc_dist = np.sum(delta_sq, axis=1) 
    delta_sq = np.square(np.asarray(normals) - gt[idx,3:])
    nml_dist = np.sum(delta_sq, axis=1)
    euc_dist = np.sqrt(euc_dist).reshape(-1,1)
    nml_dist = np.sqrt(nml_dist).reshape(-1,1)

    print('distance error: ', np.mean(euc_dist))
    print('normal error: ', np.mean(nml_dist))

def get_per_scene_errors(mesh_err_dir):
    mesh_err_files = glob.glob(f'{mesh_err_dir}/*.txt')
    mesh_err_files = [os.path.basename(f) for f in mesh_err_files]
    scenes = ['rec_Tartu1', 'rec_Tartu2', 'rec_Tartu3']
    t1d, t1n, t2d, t2n, t3d, t3n = [], [], [], [], [], []
    def read_mesh_err(mesh_err_file):
        mesh_err = np.loadtxt(mesh_err_file)
        xyz_err = (np.mean(mesh_err[:,3]))
        nml_err = (np.mean(mesh_err[:,4]))
        return xyz_err, nml_err

    for mesh_err_file in mesh_err_files:
        if mesh_err_file.startswith(scenes[0]):
            mesh_err_file = os.path.join(mesh_err_dir, mesh_err_file)
            d1, n1 = read_mesh_err(mesh_err_file)
            t1d.append(d1)
            t1n.append(n1)
        elif mesh_err_file.startswith(scenes[1]):
            mesh_err_file = os.path.join(mesh_err_dir, mesh_err_file)
            d2, n2 = read_mesh_err(mesh_err_file)
            t2d.append(d2)
            t2n.append(n2)
        elif mesh_err_file.startswith(scenes[2]):
            mesh_err_file = os.path.join(mesh_err_dir, mesh_err_file)
            d3, n3 = read_mesh_err(mesh_err_file)
            t3d.append(d3)
            t3n.append(n3)
    del mesh_err_files
    print('Tartu1 distance error: ', np.mean(t1d), 'normal error: ', np.mean(t1n))
    print('Tartu2 distance error: ', np.mean(t2d), 'normal error: ', np.mean(t2n))
    print('Tartu3 distance error: ', np.mean(t3d), 'normal error: ', np.mean(t3n))

print('Done!!')