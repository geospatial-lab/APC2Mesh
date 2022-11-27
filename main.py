import torch, os
from Config import Args as args
from Config import dtype, get_num_parts
from models.layers.mesh import Mesh, PartMesh
from base_utils import mesh_utils
import trimesh
import numpy as np
from torch import optim
from models.networks import PartNet, get_scheduler

torch.manual_seed(args.torch_seed)
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else torch.device('cpu'))
print('device: {}'.format(device))

pc_filename = os.listdir(args.partial_path)[0]  #TODO: turn this into a list for a loop over net later.
initial_xyz = np.load(os.path.join(args.partial_path, pc_filename))['unit_als']  #TODO:calculate normals for xyz in sdf_try.py

# Create the mesh.
if args.initial_mesh:
    remeshed_vertices, remeshed_faces = mesh_utils.load_obj(args.initial_mesh)
else:
    convex_hull = trimesh.convex.convex_hull(initial_xyz)
    remeshed_vertices, remeshed_faces = mesh_utils.remesh(convex_hull, 
                                                          args.initial_num_faces)
mesh_utils.save("%s/%s_initial_mesh.obj"%(args.save_path, pc_filename.split('.')[0]), remeshed_vertices, remeshed_faces)

# initial mesh
mesh = Mesh("%s/%s_initial_mesh.obj"%(args.save_path, pc_filename.split('.')[0]), device=device, hold_history=True)

input_xyz = torch.Tensor(initial_xyz).type(dtype()).to(device)[None, :, :]
# input_normals = torch.Tensor(initial_normals).type(dtype()).to(device)[None, :, :]

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
rand_verts = mesh_utils.populate_e([mesh])

print('done...!')
