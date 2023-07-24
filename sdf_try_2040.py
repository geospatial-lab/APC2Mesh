'''Data Prepation'''
from math import remainder
import os, sys
import trimesh
import numpy as np
sys.path.append('./base_utils')
from base_utils import mp_utils, file_utils, point_cloud
from mesh_to_sdf import get_surface_point_cloud
from scipy.spatial import cKDTree
import torch
from pytorch3d.ops import estimate_pointcloud_normals

def _clean_mesh(file_in, file_out, num_max_faces=None, enforce_solid=True):
    
    mesh = trimesh.load(file_in)

    mesh.process()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    if not mesh.is_watertight:
        mesh.fill_holes()
        trimesh.repair.fill_holes(mesh)

    if enforce_solid and not mesh.is_watertight:
        return

    if not mesh.is_winding_consistent:
        trimesh.repair.fix_inversion(mesh, multibody=True)
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fix_winding(mesh)

    if enforce_solid and not mesh.is_winding_consistent:
        return

    if enforce_solid and not mesh.is_volume:  # watertight, consistent winding, outward facing normals
        return

    # large meshes might cause out-of-memory errors in signed distance calculation
    if num_max_faces is None:
        mesh.export(file_out)
        print(file_out)
    elif len(mesh.faces) < num_max_faces:
        mesh.export(file_out)
        print(file_out)

def clean_meshes(in_dir, out_dir, dataset_dir, num_processes, num_max_faces=None, enforce_solid=True):
    """
    Try to repair meshes or filter broken ones. Enforce that meshes are solids to calculate signed distances.
    :param base_dir:
    :param dataset_dir:
    :param dir_in_meshes: current dir of input meshes
    :param dir_out:
    :param num_processes:
    :param num_max_faces:
    :param enforce_solid:
    :return:
    """

    dir_in_abs = os.path.join(dataset_dir, in_dir)
    dir_out_abs = os.path.join(dataset_dir, out_dir)

    os.makedirs(dir_out_abs, exist_ok=True)

    calls = []
    mesh_files = [f for f in os.listdir(dir_in_abs)
                  if os.path.isfile(os.path.join(dir_in_abs, f))]
    files_in_abs = [os.path.join(dir_in_abs, f) for f in mesh_files]
    files_out_abs = [os.path.join(dir_out_abs, f) for f in mesh_files]
    for fi, f in enumerate(mesh_files):
        if f.endswith('.mtl'):
            continue
        # skip if result already exists and is newer than the input
        if file_utils.call_necessary(files_in_abs[fi], files_out_abs[fi]):
            calls.append((files_in_abs[fi], files_out_abs[fi], num_max_faces, enforce_solid))

    mp_utils.start_process_pool(_clean_mesh, calls, num_processes)


def _normalize_mesh(file_in, file_out, trans_file_out):

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

    np.savez(trans_file_out, centroid=centroid, scale=scale) # Save to file
    # data = np.load(trans_file_out) # Load back in
    # centroid = data['centroid']; scale = data['scale']

def _translate_mesh(file_in, file_out, trans_file_out):

    mesh = trimesh.load(file_in)

    # Calculate minimum xyz
    min_xyz = mesh.vertices.min(0)

    # translate to origin
    translated_data = mesh.vertices - min_xyz

    outmesh = trimesh.Trimesh(vertices=translated_data, faces=mesh.faces)
    outmesh.export(file_out)

    np.savez(trans_file_out, min_xyz=min_xyz) # Save to file
    # data = np.load(trans_file_out) # Load back in
    # min_xyz = data['min_xyz']

def normalize_meshes(in_dir, out_dir, trans_dir, dataset_dir, num_processes=1):
    """
    Translate meshes to origin and scale to unit cube.
    :param base_dir:
    :param in_dir:
    :param trans_dir: folder where the transformation .txt files will be stored
    :param out_dir:
    :param dataset_dir:
    :param num_processes:
    :return:
    """

    in_dir_abs = os.path.join(dataset_dir, in_dir)
    out_dir_abs = os.path.join(dataset_dir, out_dir)
    trans_dir_abs = os.path.join(dataset_dir, trans_dir)

    os.makedirs(out_dir_abs, exist_ok=True)
    os.makedirs(trans_dir_abs, exist_ok=True)

    call_params = []

    mesh_files = [f for f in os.listdir(in_dir_abs)
                 if os.path.isfile(os.path.join(in_dir_abs, f))]
    temp = in_dir.split('/')[1]
    for fi, f in enumerate(mesh_files):
        ff = f'{temp}_{f}'
        if f.endswith('.mtl'):
            continue
        in_file_abs = os.path.join(in_dir_abs, f)
        out_file_abs = os.path.join(out_dir_abs, ff)
        trans_file_abs = os.path.join(trans_dir_abs, (ff[:-4]+'.npz'))

        if not file_utils.call_necessary(in_file_abs, out_file_abs):
            continue

        call_params += [(in_file_abs, out_file_abs, trans_file_abs)]

    # mp_utils.start_process_pool(_normalize_mesh, call_params, num_processes)
    mp_utils.start_process_pool(_translate_mesh, call_params, num_processes)

def denormalize(normalized_data, scale, centroid):
   # un-normalize the data
   recenteredData = np.multiply(normalized_data,scale)

   translatedPoints = recenteredData + centroid

   return(translatedPoints)

def deTranslate(translated_data, min_xyz):
    return translated_data + min_xyz

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = np.tile(np.arange(B, dtype=np.int64).reshape(view_shape),repeat_shape) #*
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,), dtype=np.int64)
    batch_indices = np.arange(B, dtype=np.int64)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3) #*
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids

def _get_sdf(file_in, file_out, fix_sample_cnt):

    mesh = trimesh.load(file_in)

    '''surf_instance contains various data, e.g. points, kd_tree, etc.'''
    surf_instance = get_surface_point_cloud(mesh, 
                                            surface_point_method='sample', 
                                            bounding_radius=1, 
                                            scan_count=30, 
                                            scan_resolution=200, 
                                            sample_point_count=40000, 
                                            calculate_normals=True)
    surf_pnt_samples = surf_instance.points
    
    '''find sdf for all surf_pnt_samples and the use fps idx to get points and corresponding sdf'''
    # use fps to reduce points to fixed number
    surf_pnt_samples = np.expand_dims(surf_pnt_samples, axis=0) # add new axis at index zero
    fps_idx = farthest_point_sample(surf_pnt_samples, fix_sample_cnt)
    fix_surf_pnts = index_points(surf_pnt_samples, fps_idx)
    fix_surf_pnts = np.squeeze(fix_surf_pnts) # add .copy() to this line and change variable name if previous line has more use down the line
    fix_sdf = surf_instance.get_sdf_in_batches(fix_surf_pnts, use_depth_buffer=False)
    np.savez(file_out, query_pnts=fix_surf_pnts, query_sdf=fix_sdf) # Save to file

def get_sdf(in_dir, out_dir, dataset_dir, fix_sample_cnt, num_processes=1):

    in_dir_abs = os.path.join(dataset_dir, in_dir)
    out_dir_abs = os.path.join(dataset_dir, out_dir)

    os.makedirs(out_dir_abs, exist_ok=True)

    call_params = []

    mesh_files = [f for f in os.listdir(in_dir_abs)
                 if os.path.isfile(os.path.join(in_dir_abs, f))]
    for fi, f in enumerate(mesh_files):
        if not f.endswith('.obj'):
            continue
        in_file_abs = os.path.join(in_dir_abs, f)
        out_file_abs = os.path.join(out_dir_abs, (f[:-4]+'.npz'))

        if not file_utils.call_necessary(in_file_abs, out_file_abs):
            continue

        call_params += [(in_file_abs, out_file_abs, fix_sample_cnt)]

    mp_utils.start_process_pool(_get_sdf, call_params, num_processes)


def _normalize_als(file_in, file_out, trsf_file):
    # Load back in
    if os.path.splitext(file_in)[1]=='.xyz':
        als_pnts = point_cloud.load_xyz(file_in)
    elif os.path.splitext(file_in)[1]=='.ply':
        als_pnts = point_cloud.load_ply(file_in)

    trsf_params = np.load(trsf_file)  
    if len(trsf_params.files) == 1:
        min_xyz = trsf_params['min_xyz']

        # translate als to mesh/sphere origin
        normalized_data = als_pnts[:,:3] - min_xyz

    elif len(trsf_params.files) == 2:

        # translate als to mesh/sphere origin
        translatedData = als_pnts[:,:3] - trsf_params['centroid']

        # scale to unit sphere
        normalized_data = np.divide(translatedData, trsf_params['scale'])

    np.savetxt(file_out, normalized_data)# Save to file i.e., XYZ

def normalize_als(in_dir, trsf_dir, out_dir, dataset_dir, num_processes=1):

    in_dir_abs = os.path.join(dataset_dir, in_dir)
    out_dir_abs = os.path.join(dataset_dir, out_dir)
    trsf_dir_abs = os.path.join(dataset_dir, trsf_dir)

    os.makedirs(out_dir_abs, exist_ok=True)

    call_params = []

    xyz_files = [f for f in os.listdir(in_dir_abs)
                 if os.path.isfile(os.path.join(in_dir_abs, f))]

    trsf_files = [os.path.splitext(f)[0] for f in os.listdir(trsf_dir_abs)
                 if os.path.isfile(os.path.join(trsf_dir_abs, f))]

    temp = in_dir.split('/')[1]
    for fi, f in enumerate(xyz_files):
        ff = f'{temp}_{f}'
        if not temp+'_'+f[:-4] in trsf_files: # if trsf doesn't, then snt should be fine too.
            print('WARNING: {}.npz is missing from the .xyz files list'.format(f[:-4]))
            continue 
        in_file_abs = os.path.join(in_dir_abs, f)
        out_file_abs = os.path.join(out_dir_abs, (ff[:-4]+'.txt'))
        trsf_file_abs = os.path.join(trsf_dir_abs, (ff[:-4]+'.npz'))

        if not file_utils.call_necessary(in_file_abs, out_file_abs):
            continue

        call_params += [(in_file_abs, out_file_abs, trsf_file_abs)]

    mp_utils.start_process_pool(_normalize_als, call_params, num_processes)


def _scale_mesh(file_in, file_out, scale): #, trans_file_out

    mesh = trimesh.load(file_in)

    # scale to desired size
    scaled_data = mesh.vertices * scale

    outmesh = trimesh.Trimesh(vertices=scaled_data, faces=mesh.faces)
    aa = outmesh.vertices + (mesh.centroid - outmesh.centroid)
    outmesh = trimesh.Trimesh(vertices=aa, faces=outmesh.faces)
    outmesh.export(file_out)

    #TODO: try _normalize with original data, n rather scale the scale


def scale_meshes(in_dir, out_dir, scale, dataset_dir, num_processes=1):
    """
    Translate meshes to origin and scale to unit cube.
    :param in_dir:
    :param out_dir:
    :param dataset_dir:
    :param num_processes:
    :return:
    """

    in_dir_abs = os.path.join(dataset_dir, in_dir)
    out_dir_abs = os.path.join(dataset_dir, out_dir)

    os.makedirs(out_dir_abs, exist_ok=True)

    call_params = []

    mesh_files = [f for f in os.listdir(in_dir_abs)
                 if os.path.isfile(os.path.join(in_dir_abs, f))]
    for fi, f in enumerate(mesh_files):
        if f.endswith('.mtl'):
            continue
        in_file_abs = os.path.join(in_dir_abs, f)
        out_file_abs = os.path.join(out_dir_abs, f)

        if not file_utils.call_necessary(in_file_abs, out_file_abs):
            continue

        call_params += [(in_file_abs, out_file_abs, scale)]

    mp_utils.start_process_pool(_scale_mesh, call_params, num_processes)

def get_clean_als(in_dir, s_mesh_dir, b_mesh_dir, out_dir, out_viz_dir, dataset_dir):
    in_dir_abs = os.path.join(dataset_dir, in_dir)
    out_dir_abs = os.path.join(dataset_dir, out_dir)
    smesh_dir_abs = os.path.join(dataset_dir, s_mesh_dir)
    bmesh_dir_abs = os.path.join(dataset_dir, b_mesh_dir)
    out_viz_dir_abs = os.path.join(dataset_dir, out_viz_dir)

    os.makedirs(out_dir_abs, exist_ok=True)
    os.makedirs(out_viz_dir_abs, exist_ok=True)

    als_files = [f for f in os.listdir(in_dir_abs)
                 if os.path.isfile(os.path.join(in_dir_abs, f))]

    for fi, f in enumerate(als_files):
        in_file_abs = os.path.join(in_dir_abs, f)
        out_file_abs = os.path.join(out_dir_abs, f)
        smesh_file_abs = os.path.join(smesh_dir_abs, (f[:-4]+'.obj'))
        bmesh_file_abs = os.path.join(bmesh_dir_abs, (f[:-4]+'.obj'))
        out_viz_file_abs = os.path.join(out_viz_dir_abs, f)

        # load point and mesh data
        als_pc = np.loadtxt(in_file_abs)
        smesh = trimesh.load(smesh_file_abs)

        '''surf_instance contains various data, e.g. points, kd_tree, etc.'''
        surf_instance = get_surface_point_cloud(smesh, 
                                                surface_point_method='sample', 
                                                bounding_radius=1, 
                                                scan_count=30, 
                                                scan_resolution=200, 
                                                sample_point_count=40000, 
                                                calculate_normals=True)

        als_sdf = surf_instance.get_sdf_in_batches(als_pc, use_depth_buffer=False, return_gradients=False)
        colors = np.zeros(als_pc.shape)
        colors[als_sdf < 0, 2] = 1
        colors[als_sdf > 0, 0] = 1

        bmesh = trimesh.load(bmesh_file_abs)
        surf_instance = get_surface_point_cloud(bmesh, 
                                                surface_point_method='sample', 
                                                bounding_radius=1, 
                                                scan_count=30, 
                                                scan_resolution=200, 
                                                sample_point_count=40000, 
                                                calculate_normals=True)

        als_sdf = surf_instance.get_sdf_in_batches(als_pc, use_depth_buffer=False, return_gradients=False)
        # colors[als_sdf < 0, 2] = 1
        colors[als_sdf > 0, 1] = 1

        als_color = np.column_stack([als_pc, colors]) 
        np.savetxt(out_viz_file_abs, als_color)

        clean_pnts = als_pc[np.where((colors[:,0] == 1) & (colors[:,1] == 0) & (colors[:,2] == 0))] 
        np.savetxt(out_file_abs, clean_pnts)


def _fix_sampling(nals_file, nmesh_file, file_out, fix_sample_cnt):

    normalized_data = np.loadtxt(nals_file) # load normalized (unit) als file 
    # subsample (down or up) ALS points to fixed count
    if len(normalized_data) > fix_sample_cnt:
        # use fps to reduce points to fixed number
        normalized_data = np.expand_dims(normalized_data, axis=0)
        fps_idx = farthest_point_sample(normalized_data , fix_sample_cnt)
        fix_pnts = index_points(normalized_data , fps_idx)
        normalized_data = np.squeeze(fix_pnts)
    elif len(normalized_data) < fix_sample_cnt:
        remainder = fix_sample_cnt - len(normalized_data)
        k = 5
        # upsample via k nearest neighbors
        while remainder != 0:
            tree = cKDTree(normalized_data)
            _, indexes = tree.query(normalized_data, k=k)

            normalized_data = np.expand_dims(normalized_data, axis=0)
            indexes = np.expand_dims(indexes, axis=0)
            bs = indexes.shape[0]
            id_0 = np.arange(bs).reshape(-1, 1, 1)
            k_groups = normalized_data[id_0, indexes]
            k_grp_centers = k_groups.mean(2).squeeze()
            normalized_data = normalized_data.squeeze()
            print(normalized_data.shape, k_grp_centers.shape)
            # noise = np.random.normal(0,0.1,[5,3])

            if (len(normalized_data) + len(k_grp_centers)) > fix_sample_cnt or (len(normalized_data) + len(k_grp_centers)) == fix_sample_cnt:
                normalized_data = np.concatenate((normalized_data, k_grp_centers[:remainder,:]), axis=0)
                print("full", normalized_data.shape)
                remainder = 0

            else:
                normalized_data = np.concatenate((normalized_data, k_grp_centers), axis=0)
                print("part", len(normalized_data))
                remainder = fix_sample_cnt - len(normalized_data)
                k = k + 3

    # get sdf of unit als
    mesh = trimesh.load(nmesh_file)

    '''surf_instance contains various data, e.g. points, kd_tree, etc.'''
    surf_instance = get_surface_point_cloud(mesh, 
                                            surface_point_method='sample', 
                                            bounding_radius=1, 
                                            scan_count=30, 
                                            scan_resolution=200, 
                                            sample_point_count=40000, 
                                            calculate_normals=True)

    als_sdf = surf_instance.get_sdf_in_batches(normalized_data, use_depth_buffer=False, return_gradients=False)

    """compute normals."""
    normals = estimate_pointcloud_normals(torch.from_numpy(np.expand_dims(normalized_data, axis=0)).float(), 15, True)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(normalized_data)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))
    normalized_data = np.column_stack([normalized_data, normals.numpy().squeeze()])

    np.savez(file_out, unit_als=normalized_data, unit_als_sdf=als_sdf)


def fix_sampling(als_dir, nmesh_dir, out_dir, dataset_dir, fix_cnt, num_processes):
    als_dir_abs = os.path.join(dataset_dir, als_dir)
    out_dir_abs = os.path.join(dataset_dir, out_dir)
    nmesh_dir_abs = os.path.join(dataset_dir, nmesh_dir)

    os.makedirs(out_dir_abs, exist_ok=True)

    call_params = []

    als_files = [f for f in os.listdir(als_dir_abs)
                 if os.path.isfile(os.path.join(als_dir_abs, f))]

    for fi, f in enumerate(als_files):
        als_file_abs = os.path.join(als_dir_abs, f)
        out_file_abs = os.path.join(out_dir_abs, f)
        nmesh_file_abs = os.path.join(nmesh_dir_abs, (f[:-4]+'.obj'))

        if not file_utils.call_necessary(als_file_abs, out_file_abs):
            continue

        call_params += [(als_file_abs, nmesh_file_abs, out_file_abs, fix_cnt)]

    mp_utils.start_process_pool(_fix_sampling, call_params, num_processes)

def main():
    dataset_dir = "/data"
    num_processes = 8
    fix_sample_cnt = 2048

    df_dir = 'all_data/original'

    mesh_list = os.listdir(f'/data/{df_dir}/mesh')
    xyz_list = os.listdir(f'/data/{df_dir}/xyz')
    print('starting file count: {}'.format(len(xyz_list)))
    for xyz_file in xyz_list:
        if xyz_file[:-4]+'.obj' not in mesh_list:
            print(xyz_file)

    print("002 Try to repair meshes or filter broken ones. Ensure solid meshes for signed distance calculations")
    # solid here means: watertight, consistent winding, outward facing normals
    clean_meshes(in_dir=f'{df_dir}/mesh', out_dir="processed/2048s/02_cleaned_ply",
                 dataset_dir=dataset_dir, num_processes=num_processes)

    print('003: scale and translate mesh, save transformation params.')
    normalize_meshes(in_dir=f'{df_dir}/mesh', out_dir='processed/2048s/03_nnt_obj', 
                    trans_dir='processed/2048s/03_trsf_npz', dataset_dir=dataset_dir, 
                    num_processes=num_processes)

    # print('004: generate complete query points set and their signed distances')
    # get_sdf(in_dir='processed/%s/03_snt_obj'%(fix_sample_cnt), 
    #         out_dir='processed/%s/04_query_npz'%(fix_sample_cnt), dataset_dir=dataset_dir, 
    #         fix_sample_cnt=fix_sample_cnt, num_processes=num_processes)

    print('005: adjust als points according to unit sphere mesh transformation params.')
    normalize_als(in_dir=f'{df_dir}/xyz', trsf_dir='processed/2048s/03_trsf_npz', 
                out_dir='processed/2048s/05_als_txt', dataset_dir=dataset_dir, 
                num_processes=num_processes)

    print('006a: scale down unit mesh.')
    scale_meshes(in_dir='processed/2048s/03_nnt_obj', 
                out_dir='processed/2048s/small03_nnt_obj', scale=0.95,
                dataset_dir=dataset_dir, num_processes=num_processes)

    print('006b: scale up unit mesh.')
    scale_meshes(in_dir='processed/2048s/03_nnt_obj', 
                out_dir='processed/2048s/big03_nnt_obj', scale=1.05,
                dataset_dir=dataset_dir, num_processes=num_processes)

    print('007: get als points outside small03_snt meshes & save to clean_als')
    get_clean_als(in_dir='processed/2048s/05_als_txt', 
                s_mesh_dir='processed/2048s/small03_nnt_obj', 
                b_mesh_dir='processed/2048s/big03_nnt_obj',
                out_dir='processed/2048s/clean_als_txt', 
                out_viz_dir='processed/2048s/outlier_viz', dataset_dir=dataset_dir)
    
    print('008: up- or down-sample clean unit als to fixed count') 
    fix_sampling(als_dir='processed/2048s/clean_als_txt',
                nmesh_dir='processed/2048s/03_nnt_obj', 
                out_dir='processed/2048s/fixed_als_txt', 
                dataset_dir=dataset_dir, fix_cnt=fix_sample_cnt, num_processes=1)

    print('done...')

if __name__ == "__main__":
    main()

    '''even after mounting volumes, u need to sudo chown -R <user:group> of non-container mount folders'''
    # os.system('cp -r /app/data/ /data/processed') # no need,fixed 