'''Data Prepation'''
import os, sys
import trimesh
import numpy as np
sys.path.append('./base_utils')
from base_utils import mp_utils, file_utils
from mesh_to_sdf import get_surface_point_cloud

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
    elif len(mesh.faces) < num_max_faces:
        mesh.export(file_out)

def clean_meshes(dataset_dir, dir_in_meshes, dir_out, num_processes, num_max_faces=None, enforce_solid=True):
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

    dir_in_abs = os.path.join("/"+dataset_dir, dir_in_meshes)
    dir_out_abs = os.path.join(dataset_dir, dir_out)

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

    # scale to unit cube
    scale = np.abs(recenteredData).max() # multiply by 2 if u want range [-0.5, 0.5]
    normalized_data = np.divide(recenteredData, scale)
    outmesh = trimesh.Trimesh(vertices=normalized_data, faces=mesh.faces)
    outmesh.export(file_out)

    np.savez(trans_file_out, centroid=centroid, scale=scale) # Save to file
    # data = np.load(trans_file_out) # Load back in
    # centroid = data['centroid']; scale = data['scale']

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

    in_dir_abs = os.path.join("/"+dataset_dir, in_dir)
    out_dir_abs = os.path.join(dataset_dir, out_dir)
    trans_dir_abs = os.path.join(dataset_dir, trans_dir)

    os.makedirs(out_dir_abs, exist_ok=True)
    os.makedirs(trans_dir_abs, exist_ok=True)

    call_params = []

    mesh_files = [f for f in os.listdir(in_dir_abs)
                 if os.path.isfile(os.path.join(in_dir_abs, f))]
    for fi, f in enumerate(mesh_files):
        if f.endswith('.mtl'):
            continue
        in_file_abs = os.path.join(in_dir_abs, f)
        out_file_abs = os.path.join(out_dir_abs, f)
        trans_file_abs = os.path.join(trans_dir_abs, (f[:-4]+'.npz'))

        if not file_utils.call_necessary(in_file_abs, out_file_abs, trans_file_abs):
            continue

        call_params += [(in_file_abs, out_file_abs, trans_file_abs)]

    mp_utils.start_process_pool(_normalize_mesh, call_params, num_processes)

def denormalize(normalized_data, scale, centroid):
   # un-normalize the data
   recenteredData = np.multiply(normalized_data,scale)

   translatedPoints = recenteredData + centroid

   return(translatedPoints)


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

def _get_sdf(file_in, file_out):

    mesh = trimesh.load(file_in)

    '''surf_instance contains various data, e.g. points, kd_tree, etc.'''
    surf_instance = get_surface_point_cloud(mesh, 
                                      surface_point_method='scan', 
                                      bounding_radius=1, 
                                      scan_count=30, 
                                      scan_resolution=200, 
                                      sample_point_count=40000, 
                                      calculate_normals=True,)
    surf_pnt_samples = surf_instance.points

    # use fps to reduce points to fixed number
    surf_pnt_samples = np.expand_dims(surf_pnt_samples, axis=0) # add new axis at index zero
    fps_idx = farthest_point_sample(surf_pnt_samples, 4096)
    fix_surf_pnts = index_points(surf_pnt_samples, fps_idx)
    fix_surf_pnts = np.squeeze(fix_surf_pnts) # add .copy() to this line and change variable name if previous line has more use down the line


def get_sdf(in_dir, out_dir, num_processes=1):

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
        out_file_abs = os.path.join(out_dir_abs, (f[:-4]+'.npy'))

        if not file_utils.call_necessary(in_file_abs, out_file_abs):
            continue

        call_params += [(in_file_abs, out_file_abs)]

    mp_utils.start_process_pool(_get_sdf, call_params, num_processes)




def main():
    dataset_dir = "data"
    num_processes = 8

    # print("002 Try to repair meshes or filter broken ones. Ensure solid meshes for signed distance calculations")
    # # solid here means: watertight, consistent winding, outward facing normals
    # clean_meshes(dataset_dir=dataset_dir, dir_in_meshes="image_1_mesh", dir_out="02_cleaned_ply", num_processes=num_processes)

    print('003 scale and translate mesh, save transformation params.')
    normalize_meshes(in_dir='image_1_mesh', out_dir='03_snt_ply', trans_dir='03_trsf_npz', dataset_dir=dataset_dir, num_processes=num_processes)

    print('004 Generate signed distances')
    get_sdf(in_dir='03_snt_ply', out_dir='04_sdf_npy',num_processes=num_processes)

if __name__ == "__main__":
    main()