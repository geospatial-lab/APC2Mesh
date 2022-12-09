import torch, os
from Config import Args as args
from Config import dtype, get_num_parts, get_num_samples
from models.layers.mesh import Mesh, PartMesh
from base_utils import mesh_utils
import trimesh
import numpy as np
from torch import optim
from models.networks import PartNet, get_scheduler
from models.losses import BeamGapLoss, chamfer_distance
import time

torch.manual_seed(args.torch_seed)
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else torch.device('cpu'))
print('device: {}'.format(device))

pc_filename = os.listdir(args.partial_path)[0]  #TODO: turn this into a list for a loop over net later.
pc_fname_no_ext = pc_filename.split('.')[0]
initial_data = np.load(os.path.join(args.partial_path, pc_filename))['unit_als']  #TODO:calculate normals for xyz in sdf_try.py
input_normals = initial_data[:, 3:]
initial_xyz = initial_data[:, :3]
del initial_data

# Create the mesh.
if args.initial_mesh:
    remeshed_vertices, remeshed_faces = mesh_utils.load_obj(args.initial_mesh)
else:
    convex_hull = trimesh.convex.convex_hull(initial_xyz)
    remeshed_vertices, remeshed_faces = mesh_utils.remesh(convex_hull,  #TODO: remove normalization, to ensure chamfer dist correctness
                                                          args.initial_num_faces)
mesh_utils.save("%s/%s_initial_mesh.obj"%(args.save_path, pc_fname_no_ext), remeshed_vertices, remeshed_faces)

# initial mesh
mesh = Mesh("%s/%s_initial_mesh.obj"%(args.save_path, pc_fname_no_ext), device=device, hold_history=True)

# normalize point cloud based on initial mesh
initial_xyz /= mesh.scale
initial_xyz += mesh.translations[None, :]
input_xyz = torch.Tensor(initial_xyz).type(dtype()).to(device)[None, :, :]
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
        xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(recon_xyz, input_xyz, 
                                                                  x_normals=recon_normals, y_normals=input_normals,
                                                                  unoriented=args.unoriented)

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

    if i % 1 == 0:
        print(f'{pc_filename}; iter: {i} out of: {args.iterations}; loss: {loss.item():.4f};'
              f' sample count: {num_samples}; time: {end_time - start_time:.2f}')

    if i % args.export_interval == 0 and i > 0:
        print('exporting reconstruction... current LR: {}'.format(optimizer.param_groups[0]['lr']))
        with torch.no_grad():
            part_mesh.export(os.path.join(args.save_path, f'{pc_fname_no_ext}recon_iter_{i}.obj'))

print('done...!')
