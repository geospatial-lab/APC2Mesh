import time, os, glob
import numpy as np

from config_a2p_il import Args as args
# from Config import Args as args
from config_p2m import dtype, get_num_parts, get_num_samples, start_logger
from post_subnets.post_ops import get_complete_files, get_dist_losses, get_per_instance_errors, config3_per_instance_error

import trimesh
import torch
from torch import optim

from base_utils import mesh_utils
from models.layers.mesh import Mesh, PartMesh
from models.networks_p2m import PartNet, get_scheduler
from models.losses import BeamGapLoss, chamfer_distance, point_mesh_loss
from pytorch3d.ops.points_normals import estimate_pointcloud_normals


def compute_normals(pc):
    pc = pc.reshape(-1,3)
    pc = torch.from_numpy(pc).to(device)
    pc = pc.unsqueeze(0)
    pc = pc.type(dtype())
    normals = estimate_pointcloud_normals(pc, 12)
    normals = normals.squeeze().cpu().numpy()
    return normals


torch.manual_seed(args.torch_seed)
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else torch.device('cpu'))
print('device: {}'.format(device))

# files and dirs prep
exp_num = '0'
# subdirs = ['als', 'twentyFivePCC', 'fiftyPCC'] # 'seventyFivePCC'
# for subdir in subdirs:
#     if subdir == subdirs[0]:
#         exp_num = 11
#     elif subdir == subdirs[1]:
#         exp_num = 10
#     elif subdir == subdirs[2]:
#         exp_num = 9

os.makedirs(os.path.join(args.save_path,'config-a2pil{}'.format(exp_num)), exist_ok=True)  # config details are in matching loss_log file.

# start logger
p2m_logger = start_logger(log_dir=args.p2m_logs, fname='loss_log-a2pil{}'.format(exp_num))
p2m_logger.info('description: For comparative analysis in A2P_IL work') #  - {}'.format(subdir))
p2m_logger.info(f'Attention: {args.attention}') # @ feat_dim {96}')
p2m_logger.info('init#faces: {}'.format(args.initial_num_faces))
p2m_logger.info('init_samples: {}'.format(args.init_samples))
p2m_logger.info('iterations & upsampling: {} & {}'.format(args.iterations, args.upsamp))
p2m_logger.info('normals ang wt: {}  |  local non-uniform weight: {}'.format(args.ang_wt, args.local_non_uniform))
p2m_logger.info('beamgap iterations & modulo: {} | {}'.format(args.beamgap_iterations, args.beamgap_modulo))
p2m_logger.info('normals orientation: {} \n\n'.format(args.unoriented))

#retrieve individual completed point files from the batched .npz files 
# if not os.path.exists(f'{args.data_path}/als') or not os.listdir(f'{args.data_path}/als'):
#     pcc_blist = glob.glob(f'{args.pcc_npz_dir}/*.npz')
#     get_complete_files(blist=pcc_blist, rt_dir=args.pcc_npz_dir, save_dir=args.data_path)

# get all the files in the data_path
rebt_files = glob.glob(f'{args.data_path}/*.txt')
for i, pc_filepath in enumerate(rebt_files):
    # pc_folder = pc_filepath.split('/')[-2]
    # if not pc_folder == 'fine': # if pc_folder == 'gt': 
    #     continue
    rebt_files[i] = pc_filepath.split('/')[-1] # pc_folder +'/'+pc_filepath.split('/')[-1]

# for pc_filename in os.listdir(f'{args.data_path}/{subdir}'): # fine, als, subdir
for pc_filename in rebt_files:
    # pc_filename = 'Tartu1_4770.txt'
    # pc_folder = pc_filename.split('/')[0]
    pc_fname_no_ext = pc_filename.split('.')[0]
    file_chk = os.path.join(args.save_path, f'config-a2pil{exp_num}', f'last_rec_{pc_fname_no_ext}.obj')

    if os.path.exists(file_chk):
        print(f'file {pc_filename} already done, skipping...')
        continue
    
    p2m_logger.info('\n** ' + pc_fname_no_ext)
    # initial_data = np.loadtxt(os.path.join(f'{args.data_path}/{subdir}', pc_filename))  #TODO:save normals for pcc_out files in main_pcc.py
    initial_data = np.loadtxt(os.path.join(f'{args.data_path}', pc_filename))
    # initial_data = np.loadtxt(os.path.join(f'{args.data_path}/fine', pc_filename))  #TODO:save normals for pcc_out files in main_pcc.py
    if initial_data.shape[1] == 3:
        initial_data = np.hstack((initial_data, compute_normals(initial_data)))
    input_normals = initial_data[:, 3:]
    # input_normals = pcc_as_gt[:, 3:]
    # pcc_xyz_as_gt = pcc_as_gt[:, :3]
    initial_xyz = initial_data[:, :3]
    del initial_data

    # Create the mesh.
    if args.initial_mesh:
        remeshed_vertices, remeshed_faces = mesh_utils.load_obj(args.initial_mesh)
    else:
        convex_hull = trimesh.convex.convex_hull(initial_xyz)
        remeshed_vertices, remeshed_faces = mesh_utils.remesh(convex_hull,  #TODO: remove normalization, to ensure chamfer dist coa2pilectness
                                                            args.initial_num_faces)
    mesh_utils.save("%s/%s_initial_mesh.obj"%(os.path.join(args.save_path, f'config-a2pil{exp_num}'), pc_fname_no_ext), remeshed_vertices, remeshed_faces)

    # initial mesh
    mesh = Mesh("%s/%s_initial_mesh.obj"%(os.path.join(args.save_path, f'config-a2pil{exp_num}'), pc_fname_no_ext), device=device, hold_history=True)

    # normalize point cloud based on initial mesh
    initial_xyz /= mesh.scale
    initial_xyz += mesh.translations[None, :]
    input_xyz = torch.Tensor(initial_xyz).type(dtype()).to(device)[None, :, :]

    # pcc_xyz_as_gt /= mesh.scale
    # pcc_xyz_as_gt += mesh.translations[None, :]
    # pcc_xyz_as_gt = torch.Tensor(pcc_xyz_as_gt).type(dtype()).to(device)[None, :, :]

    input_normals = torch.Tensor(input_normals).type(dtype()).to(device)[None, :, :]

    part_mesh = PartMesh(mesh, num_parts=get_num_parts(args,len(mesh.faces)), bfs_depth=args.overlap)
    print(f'number of parts {part_mesh.n_submeshes}')

    # initialize network, weights, and random input tensor
    init_verts = mesh.vs.clone().detach()
    model = PartNet(init_part_mesh=part_mesh, convs=args.convs,
                    pool=args.pools, res_blocks=args.res_blks,
                    init_verts=init_verts, transfer_data=args.transfer_data,
                    leaky=args.lrelu_alpha, init_weights_size=args.init_weights).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(args.iterations, optimizer)
    rand_verts = mesh_utils.populate_e([mesh])  # totally random verticies re-indexed with mesh edges. NB: 6 = a pair 3D random coords of an egde  

    beamgap_loss = BeamGapLoss(device)
    if args.beamgap_iterations > 0:
        print('beamgap on')
        beamgap_loss.update_pm(part_mesh, torch.cat([input_xyz, input_normals], dim=-1))

    for i in range(args.iterations):
        num_samples = get_num_samples(args, i % args.upsamp)
        if args.global_step:  # only matters if part_mesh.n_submeshes > 2
            optimizer.zero_grad()
        start_time = time.time()
        for part_i, est_verts in enumerate(model(rand_verts, part_mesh)):
            if not args.global_step:
                optimizer.zero_grad()
            part_mesh.update_verts(est_verts[0], part_i)
            num_samples = get_num_samples(args, i % args.upsamp)
            recon_xyz, recon_normals = mesh_utils.sample_surface(part_mesh.main_mesh.faces, part_mesh.main_mesh.vs.unsqueeze(0), num_samples)
            
            # calc chamfer loss w/ normals
            recon_xyz, recon_normals = recon_xyz.type(dtype()), recon_normals.type(dtype())
            xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(recon_xyz, input_xyz, # replaced input_xyz with pcc_xyz_as_gt
                                                                    x_normals=recon_normals, y_normals=input_normals,
                                                                    unoriented=args.unoriented)
            # # calc point to mesh loss
            # pnt_face_loss = point_mesh_loss(input_xyz, part_mesh.main_mesh)
            # loss = ((args.ang_wt * normals_chamfer_loss) + pnt_face_loss)  
            if (i < args.beamgap_iterations) and (i % args.beamgap_modulo == 0):
                loss = beamgap_loss(part_mesh, part_i)
            else:
                loss = (xyz_chamfer_loss + (args.ang_wt * normals_chamfer_loss))  # this n the next loss component might be related to normal consistency loss
            if args.local_non_uniform > 0:
                loss += args.local_non_uniform * mesh_utils.local_nonuniform_penalty(part_mesh.main_mesh).float()
            loss.backward()

            if not args.global_step:
                optimizer.step()
                scheduler.step()
            part_mesh.main_mesh.vs.detach_()

        if args.global_step:
            optimizer.step()
            scheduler.step()

        end_time = time.time()

        if i % 4 == 0:
            p2m_logger.info(f'{pc_filename}; iter: {i:4d} out of: {args.iterations}; loss: {loss.item():.4f};'
                f' sample count: {num_samples}; time: {end_time - start_time:.2f}')

        if i % args.export_interval == 0 and i > 0:
            print('exporting reconstruction... current LR: {}'.format(optimizer.param_groups[0]['lr']))
            with torch.no_grad():
                part_mesh.export(os.path.join(args.save_path, f'config-a2pil{exp_num}', f'rec_{pc_fname_no_ext}.obj'))
            mesh_path = os.path.join(args.save_path, f'config-a2pil{exp_num}', f'rec_{pc_fname_no_ext}.obj')
            # get_per_instance_errors(f'{args.data_path}/gt', pc_fname_no_ext, mesh_path)
            # perr, derr = config3_per_instance_error(pc_fname_no_ext)

        if (i > 0 and (i + 1) % args.upsamp == 0):
            mesh = part_mesh.main_mesh
            num_faces = int(np.clip(len(mesh.faces) * 1.5, len(mesh.faces), args.max_faces))

            if num_faces > len(mesh.faces) or args.manifold_always:
                # up-sample mesh
                mesh = mesh_utils.manifold_upsample(mesh, f'{args.save_path}/config-a2pil{exp_num}', pc_fname_no_ext, Mesh,
                                            num_faces=min(num_faces, args.max_faces),
                                            res=args.manifold_res, simplify=True)

                part_mesh = PartMesh(mesh, num_parts=get_num_parts(args,len(mesh.faces)), bfs_depth=args.overlap)
                print(f'upsampled to {len(mesh.faces)} faces; number of parts {part_mesh.n_submeshes}')
                
                # re-initialize the network and it params to re-fit the upsampled mesh 
                init_verts = mesh.vs.clone().detach()
                model = PartNet(init_part_mesh=part_mesh, convs=args.convs,
                                pool=args.pools, res_blocks=args.res_blks,
                                init_verts=init_verts, transfer_data=args.transfer_data,
                                leaky=args.lrelu_alpha, init_weights_size=args.init_weights).to(device)

                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                scheduler = get_scheduler(args.iterations, optimizer)
                rand_verts = mesh_utils.populate_e([mesh])  # totally random verticies re-indexed with mesh edges. NB: 6 = a pair 3D random coords of an egde  

                if i < args.beamgap_iterations:
                    print('beamgap updated')
                    beamgap_loss.update_pm(part_mesh, input_xyz)

    p2m_logger.info(f'{pc_filename}; final chamfer xyz loss: {xyz_chamfer_loss.item():.7f};')
    with torch.no_grad():
        mesh.export(os.path.join(args.save_path, f'config-a2pil{exp_num}', f'last_rec_{pc_fname_no_ext}.obj'))

print('p2m done...!')

# compute final losses
rlist = glob.glob(os.path.join(args.save_path, f'config-a2pil{exp_num}/rec_*.obj'))
glist = glob.glob(f'{args.data_path}/gt/*.txt')
get_dist_losses(rec_list=rlist, gt_list=glist) #FIXME: check for scale/translation consistency

print('losses done...!')
