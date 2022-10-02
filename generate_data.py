import os, sys
import trimesh
import shutil
sys.path.append('./base_utils')
import numpy as np
import trimesh.transformations as trafo
from base_utils import mp_utils, file_utils, utils, point_cloud


def _convert_mesh(in_mesh, out_mesh):

    mesh = None
    try:
        mesh = trimesh.load(in_mesh)
    except AttributeError as e:
        print(e)
    except IndexError as e:
        print(e)
    except ValueError as e:
        print(e)
    except NameError as e:
        print(e)

    if mesh is not None:
        try:
            mesh.export(out_mesh)
        except ValueError as e:
            print(e)


def convert_meshes(in_dir_abs, out_dir_abs, target_file_type: str, num_processes=8):
    """
    Convert a mesh file to another file type.
    :param in_dir_abs:
    :param out_dir_abs:
    :param target_file_type: ending of wanted mesh file, e.g. '.ply'
    :return:
    """

    os.makedirs(out_dir_abs, exist_ok=True)

    mesh_files = []
    for root, dirs, files in os.walk(in_dir_abs, topdown=True):
        for name in files:
            mesh_files.append(os.path.join(root, name))

    allowed_mesh_types = ['.off', '.ply', '.obj', '.stl']
    mesh_files = list(filter(lambda f: (f[-4:] in allowed_mesh_types), mesh_files))

    calls = []
    for fi, f in enumerate(mesh_files):
        file_base_name = os.path.basename(f)
        file_out = os.path.join(out_dir_abs, file_base_name[:-4] + target_file_type) # replace extension and place in out folder
        if file_utils.call_necessary(f, file_out):
            calls.append((f, file_out))

    mp_utils.start_process_pool(_convert_mesh, calls, num_processes)
    
    # determine the number of objects generated compared to the original count
    print("In_dir count: {} -- Out_dir count: {}".format(len(os.listdir(in_dir_abs)), len(os.listdir(out_dir_abs))))


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

    dir_in_abs = os.path.join(dataset_dir, dir_in_meshes)
    dir_out_abs = os.path.join(dataset_dir, dir_out)

    os.makedirs(dir_out_abs, exist_ok=True)

    calls = []
    mesh_files = [f for f in os.listdir(dir_in_abs)
                  if os.path.isfile(os.path.join(dir_in_abs, f))]
    files_in_abs = [os.path.join(dir_in_abs, f) for f in mesh_files]
    files_out_abs = [os.path.join(dir_out_abs, f) for f in mesh_files]
    for fi, f in enumerate(mesh_files):
        # skip if result already exists and is newer than the input
        if file_utils.call_necessary(files_in_abs[fi], files_out_abs[fi]):
            calls.append((files_in_abs[fi], files_out_abs[fi], num_max_faces, enforce_solid))

    mp_utils.start_process_pool(_clean_mesh, calls, num_processes)

    # determine the number of objects generated compared to the original count
    print("In_dir count: {} -- Out_dir count: {}".format(len(os.listdir(dir_in_abs)), len(os.listdir(dir_out_abs))))


def _normalize_mesh(file_in, file_out, denorm_file_out):

    mesh = trimesh.load(file_in)

    # Find the centroid
    centroid = mesh.vertices.mean(0)
    
    # subrtact centroid from the data
    recenteredData = mesh.vertices - centroid

    # Calculate Scale factor
    scale = np.abs(recenteredData).max()*2
    # Normalize
    normalized_data = np.divide(recenteredData,scale)

    mesh.vertices = normalized_data

    # print("assert mesh.vertices is replaced by normalized_data: {}".format(np.allclose(mesh.vertices, normalized_data)))

    # save transformation parameters for denormalization later
    np.savetxt(denorm_file_out, (scale, centroid), fmt='%s')

    # bounds = mesh.extents
    # if bounds.min() == 0.0:
    #     return

    # # translate to origin
    # translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    # translation = trimesh.transformations.translation_matrix(direction=-translation)
    # mesh.apply_transform(translation)

    # # scale to unit cube
    # scale = 1.0/bounds.max()
    # scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    # mesh.apply_transform(scale_trafo)

    mesh.export(file_out)


def normalize_meshes(in_dir, out_dir, out_dir_denorm, dataset_dir, num_processes=1):
    """
    Translate meshes to origin and scale to unit cube.
    :param base_dir:
    :param in_dir:
    :param filter_dir:
    :param out_dir:
    :param denorm: transformation parameters files for denormalization of unit cube meshes
    :param dataset_dir:
    :param num_processes:
    :return:
    """

    in_dir_abs = os.path.join(dataset_dir, in_dir)
    out_dir_abs = os.path.join(dataset_dir, out_dir)
    out_dir_denorm_abs = os.path.join(dataset_dir, out_dir_denorm)

    os.makedirs(out_dir_abs, exist_ok=True)
    os.makedirs(out_dir_denorm_abs, exist_ok=True)

    call_params = []

    mesh_files = [f for f in os.listdir(in_dir_abs)
                 if os.path.isfile(os.path.join(in_dir_abs, f))]
    for fi, f in enumerate(mesh_files):
        in_file_abs = os.path.join(in_dir_abs, f)
        out_file_abs = os.path.join(out_dir_abs, f)
        file_base_name = os.path.basename(f)
        out_denorm_file_abs = os.path.join(out_dir_denorm_abs, file_base_name[:-4] + ".txt")

        if not file_utils.call_necessary(in_file_abs, out_file_abs):
            continue

        call_params += [(in_file_abs, out_file_abs, out_denorm_file_abs)]

    mp_utils.start_process_pool(_normalize_mesh, call_params, num_processes)


def _pcd_files_to_pts(pcd_files, pts_file_npy, pts_file, obj_locations, obj_rotations, min_pts_size=0, debug=False):
    """
    Convert pcd blensor results to xyz or directly to npy files. Merge front and back scans.
    Moving the object instead of the camera because the point cloud is in some very weird space that behaves
    crazy when the camera moves. A full day wasted on this shit!
    :param pcd_files:
    :param pts_file_npy:
    :param pts_file:
    :param trafos_inv:
    :param debug:
    :return:
    """

    import gzip

    def revert_offset(pts_data: np.ndarray, inv_offset: np.ndarray):
        pts_reverted = pts_data
        # don't just check the header because missing rays may be added with NaNs
        if pts_reverted.shape[0] > 0:
            pts_offset_correction = np.broadcast_to(inv_offset, pts_reverted.shape)
            pts_reverted += pts_offset_correction

        return pts_reverted

    # https://www.blensor.org/numpy_import.html
    def extract_xyz_from_blensor_numpy(arr_raw):
        # timestamp
        # yaw, pitch
        # distance,distance_noise
        # x,y,z
        # x_noise,y_noise,z_noise
        # object_id
        # 255*color[0]
        # 255*color[1]
        # 255*color[2]
        # idx
        hits = arr_raw[arr_raw[:, 3] != 0.0]  # distance != 0.0 --> hit
        noisy_xyz = hits[:, [8, 9, 10]]
        return noisy_xyz

    pts_data_to_cat = []
    for fi, f in enumerate(pcd_files):
        try:
            if f.endswith('.numpy.gz'):
                pts_data_vs = extract_xyz_from_blensor_numpy(np.loadtxt(gzip.GzipFile(f, "r")))
            elif f.endswith('.numpy'):
                pts_data_vs = extract_xyz_from_blensor_numpy(np.loadtxt(f))
            elif f.endswith('.pcd'):
                pts_data_vs, header_info = point_cloud.load_pcd(file_in=f)
            else:
                raise ValueError('Input file {} has an unknown format!'.format(f))
        except EOFError as er:
            print('Error processing {}: {}'.format(f, er))
            continue

        # undo coordinate system changes
        pts_data_vs = utils.right_handed_to_left_handed(pts_data_vs)

        # move back from camera distance, always along x axis
        obj_location = np.array(obj_locations[fi])
        revert_offset(pts_data_vs, -obj_location)

        # get and apply inverse rotation matrix of camera
        scanner_rotation_inv = trafo.quaternion_matrix(trafo.quaternion_conjugate(obj_rotations[fi]))
        pts_data_ws_test_inv = trafo.transform_points(pts_data_vs, scanner_rotation_inv, translate=False)
        pts_data_ws = pts_data_ws_test_inv

        if pts_data_ws.shape[0] > 0:
            pts_data_to_cat += [pts_data_ws.astype(np.float32)]

        # debug outputs to check the rotations... the pointcloud MUST align exactly with the mesh
        if debug:
            point_cloud.write_xyz(file_path=os.path.join('debug', 'test_{}.xyz'.format(str(fi))), points=pts_data_ws)

    if len(pts_data_to_cat) > 0:
        pts_data = np.concatenate(tuple(pts_data_to_cat), axis=0)

        if pts_data.shape[0] > min_pts_size:
            point_cloud.write_xyz(file_path=pts_file, points=pts_data)
            np.save(pts_file_npy, pts_data)


def sample_blensor(dataset_dir, blensor_bin, dir_in,
                   dir_out, dir_out_vis, dir_out_pcd, dir_blensor_scripts,
                   num_scans_per_mesh_min, num_scans_per_mesh_max, num_processes, min_pts_size=0,
                   scanner_noise_sigma_min=0.0, scanner_noise_sigma_max=0.05):
    """
    Call Blender to use a Blensor script to sample a point cloud from a mesh
    :param base_dir:
    :param dataset_dir:
    :param dir_in:
    :param dir_out:
    :param dir_blensor_scripts:
    :param num_scans_per_mesh_min: default: 5
    :param num_scans_per_mesh_max: default: 100
    :param scanner_noise_sigma_min: default: 0.0004, rather a lot: 0.01
    :param scanner_noise_sigma_max: default: 0.0004, rather a lot: 0.01
    :return:
    """

    # test blensor scripts with: .\blender -P 00990000_6216c8dabde0a997e09b0f42_trimesh_000.py

    # blender_path = os.path.join(base_dir, blensor_bin)
    blender_path = blensor_bin
    dir_abs_in = os.path.join(dataset_dir, dir_in)
    dir_abs_out = os.path.join(dataset_dir, dir_out)
    dir_abs_out_vis = os.path.join(dataset_dir, dir_out_vis)
    dir_abs_blensor = os.path.join(dataset_dir, dir_blensor_scripts)
    dir_abs_pcd = os.path.join(dataset_dir, dir_out_pcd)

    os.makedirs(dir_abs_out, exist_ok=True)
    os.makedirs(dir_abs_out_vis, exist_ok=True)
    os.makedirs(dir_abs_blensor, exist_ok=True)
    os.makedirs(dir_abs_pcd, exist_ok=True)

    with open('blensor_script_template.py', 'r') as file:
        blensor_script_template = file.read()

    blender_blensor_calls = []
    pcd_base_files = []
    pcd_noisy_files = []
    obj_locations = []
    obj_rotations = []

    obj_files = [f for f in os.listdir(dir_abs_in)
                 if os.path.isfile(os.path.join(dir_abs_in, f)) and f[-4:] == '.ply']
    for fi, file in enumerate(obj_files):

        # gather all file names involved in the blensor scanning
        obj_file = os.path.join(dir_abs_in, file)
        blensor_script_file = os.path.join(dir_abs_blensor, file[:-4] + '.py')

        new_pcd_base_files = []
        new_pcd_noisy_files = []
        new_obj_locations = []
        new_obj_rotations = []
        rnd = np.random.RandomState(file_utils.filename_to_hash(obj_file))
        num_scans = rnd.randint(num_scans_per_mesh_min, num_scans_per_mesh_max + 1)
        noise_sigma = rnd.rand() * (scanner_noise_sigma_max - scanner_noise_sigma_min) + scanner_noise_sigma_min
        for num_scan in range(num_scans):
            pcd_base_file = os.path.join(
                dir_abs_pcd, file[:-4] + '_{num}.numpy.gz'.format(num=str(num_scan).zfill(5)))
            pcd_noisy_file = pcd_base_file[:-9] + '00000.numpy.gz'

            obj_location = (rnd.rand(3) * 2.0 - 1.0)
            obj_location_rand_factors = np.array([0.1, 1.0, 0.1])
            obj_location *= obj_location_rand_factors
            obj_location[1] += 4.0  # offset in cam view dir
            obj_rotation = trafo.random_quaternion(rnd.rand(3))

            # extend lists of pcd output files
            new_pcd_base_files.append(pcd_base_file)
            new_pcd_noisy_files.append(pcd_noisy_file)
            new_obj_locations.append(obj_location.tolist())
            new_obj_rotations.append(obj_rotation.tolist())

        new_scan_sigmas = [noise_sigma] * num_scans

        pcd_base_files.append(new_pcd_base_files)
        pcd_noisy_files.append(new_pcd_noisy_files)
        obj_locations.append(new_obj_locations)
        obj_rotations.append(new_obj_rotations)

        # prepare blensor calls if necessary
        output_files = [os.path.join(dir_abs_pcd, os.path.basename(f)) for f in new_pcd_noisy_files]
        output_files += [blensor_script_file]
        if file_utils.call_necessary(obj_file, output_files):
            blensor_script = blensor_script_template.format(
                file_loc=obj_file,
                obj_locations=str(new_obj_locations),
                obj_rotations=str(new_obj_rotations),
                evd_files=str(new_pcd_base_files),
                scan_sigmas=str(new_scan_sigmas),
            )
            blensor_script = blensor_script.replace('\\', '/')  # '\' would require escape sequence

            with open(blensor_script_file, "w") as text_file:
                text_file.write(blensor_script)

            # start blender with python script (-P) and close without prompt (-b)
            blender_blensor_call = '{} -P {} -b'.format(blender_path, blensor_script_file)
            blender_blensor_calls.append((blender_blensor_call,))

    mp_utils.start_process_pool(mp_utils.mp_worker, blender_blensor_calls, num_processes)

    def get_pcd_origin_file(pcd_file):
        origin_file = os.path.basename(pcd_file)[:-9] + '.xyz'
        origin_file = origin_file.replace('00000.xyz', '.xyz')
        origin_file = origin_file.replace('_noisy.xyz', '.xyz')
        origin_file = origin_file.replace('_00000.xyz', '.xyz')
        return origin_file

    print('### convert pcd to pts')
    call_params = []
    for fi, files in enumerate(pcd_noisy_files):
        pcd_files_abs = [os.path.join(dir_abs_pcd, os.path.basename(f)) for f in files]
        pcd_origin = get_pcd_origin_file(files[0])
        xyz_file = os.path.join(dir_abs_out_vis, pcd_origin)
        xyz_npy_file = os.path.join(dir_abs_out, pcd_origin + '.npy')

        if file_utils.call_necessary(pcd_files_abs, [xyz_npy_file, xyz_file]):
            call_params += [(pcd_files_abs, xyz_npy_file, xyz_file, obj_locations[fi], obj_rotations[fi], min_pts_size)]

    mp_utils.start_process_pool(_pcd_files_to_pts, call_params, num_processes)

    # determine the number of objects generated compared to the original count
    print("In_dir count: {} -- Out_dir count: {}".format(len(os.listdir(dir_abs_in)), len(os.listdir(dir_abs_out_vis))))


def make_dataset_splits(dataset_dir, final_out_dir, seed=42, only_test_set=False, only_test_set_dir=None, testset_ratio=0.2):

    """:params: only_test_set: all the files in final_out_dir will be used for testing"""

    import random
    rnd = random.Random(seed)

    # write files for train / test / eval set
    final_out_dir_abs = os.path.join(dataset_dir, final_out_dir)
    final_output_files = [f for f in os.listdir(final_out_dir_abs)
                          if os.path.isfile(os.path.join(final_out_dir_abs, f)) and f[-4:] == '.npy']
    files_dataset = [f[:-8] for f in final_output_files]

    if len(files_dataset) == 0:
        raise ValueError('Dataset is empty! {}'.format(final_out_dir_abs))

    if only_test_set and only_test_set_dir is not None:
        files_test = files_dataset
    else:
        # files_test = rnd.sample(files_dataset, max(3, min(int(testset_ratio * len(files_dataset)), 100)))  # 3..50, ~10%
        files_test = rnd.sample(files_dataset, int(len(files_dataset) * testset_ratio))
    files_train = list(set(files_dataset).difference(set(files_test)))

    files_test.sort()
    files_train.sort()

    file_train_set = os.path.join(dataset_dir, 'trainset.txt')
    file_test_set = os.path.join(dataset_dir, 'testset.txt')
    file_val_set = os.path.join(dataset_dir, 'valset.txt')

    file_utils.make_dir_for_file(file_test_set)
    nl = '\n'
    file_test_set_str = nl.join(files_test)
    file_train_set_str = nl.join(files_train)
    file_val_set_str = nl.join(files_test[:100])
    with open(file_test_set, "w") as text_file:
        text_file.write(file_test_set_str)
    if not only_test_set:
        with open(file_train_set, "w") as text_file:
            text_file.write(file_train_set_str)
    with open(file_val_set, "w") as text_file:
        text_file.write(file_val_set_str)  # validate the test set by default


def clean_up_broken_inputs(dataset_dir, final_out_dir, final_out_extension, clean_up_dirs, broken_dir='broken'):
    """
    Assume that the file stem (excluding path and everything after the first '.') is a unique identifier in
    multiple directories.

    :param dataset_dir:
    :param final_out_dir: the last dir with eliminated files
    :param clean_up_dirs: list of dirs to cleans in order to match the contents of "final_out_dir"
    :return:
    """

    final_out_dir_abs = os.path.join(dataset_dir, final_out_dir)
    final_output_files = [f for f in os.listdir(final_out_dir_abs)
                          if os.path.isfile(os.path.join(final_out_dir_abs, f)) and
                          (final_out_extension is None or f[-len(final_out_extension):] == final_out_extension)]

    if len(final_output_files) == 0:
        print('Warning: Output dir "{}" is empty'.format(final_out_dir_abs))
        return

    # move inputs and intermediate results that have no final output
    final_output_file_stems = set(tuple([f.split('.', 1)[0] for f in final_output_files]))
    # final_output_file_stem_lengths = [len(f.split('.', 1)[0]) for f in final_output_files]
    # num_final_output_file_stem_lengths = len(set(final_output_file_stem_lengths))
    # inconsistent_file_length = num_final_output_file_stem_lengths > 1
    # if inconsistent_file_length:
    #     print('WARNING: output files don\'t have consistent length. Clean-up broken inputs may do unwanted things.')
    for clean_up_dir in clean_up_dirs:
        dir_abs = os.path.join(dataset_dir, clean_up_dir)
        if not os.path.isdir(dir_abs):
            continue
        dir_files = [f for f in os.listdir(dir_abs) if os.path.isfile(os.path.join(dir_abs, f))]
        dir_file_stems = [f.split('.', 1)[0] for f in dir_files]
        dir_file_stems_without_final_output = [f not in final_output_file_stems for f in dir_file_stems]
        dir_files_without_final_output = np.array(dir_files)[dir_file_stems_without_final_output]

        broken_dir_abs = os.path.join(dataset_dir, broken_dir, clean_up_dir)
        broken_files = [os.path.join(broken_dir_abs, f) for f in dir_files_without_final_output]

        for fi, f in enumerate(dir_files_without_final_output):
            os.makedirs(broken_dir_abs, exist_ok=True)
            ext = f.split('.', 1)[1]
            if ext != "mtl" or ext != "prj":
                shutil.move(os.path.join(dir_abs, f), broken_files[fi])


def main():

    blensor_bin = "bin/Blensor-x64.AppImage"
    num_processes = 8

    only_for_evaluation = False
    num_scans_per_mesh_min = 5
    num_scans_per_mesh_max = 30
    scanner_noise_sigma_min = 0.0004
    scanner_noise_sigma_max = 0.01


    dataset_list = ["Harku"] # , "Kiili"
    dirs_to_clean = ["00_data_obj", "01_data_ply", "02_cleaned_ply", "03_denorm", "03_snt_ply",
                     "04_blensor_py", "04_pcd"]

    for d in dataset_list:

        dataset_dir = f"/home/data/test_case/00_data/{d}"

        # print('001 convert base meshes to ply')
        # convert_meshes(in_dir_abs=os.path.join(dataset_dir, "00_data_obj"), out_dir_abs=os.path.join(dataset_dir, "01_data_ply"), target_file_type=".ply")
        
        # print("002 Try to repair meshes or filter broken ones. Ensure solid meshes for signed distance calculations")
        # # solid here means: watertight, consistent winding, outward facing normals
        # clean_meshes(dataset_dir=dataset_dir, dir_in_meshes="01_data_ply", dir_out="02_cleaned_ply", num_processes=num_processes)

        # print("002a filter out none-solid meshes")  # Harku: 14167 --> 13995
        # clean_up_broken_inputs(dataset_dir=dataset_dir, final_out_dir='02_cleaned_ply', final_out_extension='.ply',
        #                        clean_up_dirs=dirs_to_clean, broken_dir='broken')

        # print('003 scale and translate mesh')
        # normalize_meshes(in_dir='02_cleaned_ply', out_dir='03_snt_ply', out_dir_denorm='03_denorm', dataset_dir=dataset_dir, num_processes=num_processes)
    
        # print('004 sample with Blensor')
        # sample_blensor(dataset_dir=dataset_dir, blensor_bin=blensor_bin,
        #             dir_in='03_snt_ply', dir_out='04_pts', dir_out_vis='04_pts_vis', dir_out_pcd='04_pcd', dir_blensor_scripts='04_blensor_py',
        #             num_scans_per_mesh_min=num_scans_per_mesh_min, num_scans_per_mesh_max=num_scans_per_mesh_max,
        #             num_processes=num_processes, min_pts_size=0 if only_for_evaluation else 5000,
        #             scanner_noise_sigma_min=scanner_noise_sigma_min, scanner_noise_sigma_max=scanner_noise_sigma_max)

        # print("004a filter out files abandoned in 'sample_blensor'")
        # # NOTE: this will move the "04_pcd" folder to the "broken" folder
        # clean_up_broken_inputs(dataset_dir=dataset_dir, final_out_dir="04_pts_vis", final_out_extension='.xyz', 
        #                        clean_up_dirs=dirs_to_clean, broken_dir='broken')

        # print("005 split point sets into train / val / test")
        # make_dataset_splits(dataset_dir=dataset_dir, final_out_dir="04_pts")

        print('*** convert 03_snt_ply mesh from .ply to .obj for IntersectionXYZpn')
        convert_meshes(in_dir_abs=os.path.join(dataset_dir, "03_snt_ply"), out_dir_abs=os.path.join(dataset_dir, "03_snt_obj"), target_file_type=".obj")


if __name__ == "__main__":
    main()