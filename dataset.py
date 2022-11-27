import os
import numpy as np
import trimesh
import uuid
import torch

MANIFOLD_SOFTWARE_DIR = "/Manifold/build"


def random_file_name(ext, prefix="tmp_"):
    return f"{prefix}{uuid.uuid4()}.{ext}"


def remesh(hull_mesh, num_faces):

    # Write the original mesh as OBJ.
    original_file = random_file_name("obj")
    with open(original_file, "w") as f:
        mesh = trimesh.Trimesh(vertices=hull_mesh.vertices, faces=hull_mesh.faces)
        f.write(trimesh.exchange.obj.export_obj(mesh))

    # Create a manifold of the original file.
    manifold_file = random_file_name("obj")
    manifold_script_path = os.path.join(MANIFOLD_SOFTWARE_DIR, "manifold")
    cmd = f"{manifold_script_path} {original_file} {manifold_file}"
    os.system(cmd)

    # Simplify the manifold.
    simplified_file = random_file_name("obj")
    simplify_script_path = os.path.join(MANIFOLD_SOFTWARE_DIR, "simplify")
    cmd = (
        f"{simplify_script_path} -i {manifold_file} -o {simplified_file} -f {num_faces}"
    )
    os.system(cmd)

    # Read the simplified manifold.
    with open(simplified_file, "r") as f:
        mesh = trimesh.exchange.obj.load_obj(f)

    # Prevent file spam.
    os.remove(original_file)
    os.remove(manifold_file)
    os.remove(simplified_file)

    return mesh["vertices"], mesh["faces"]


data_path = '/data/processed/%d' %(4096)
complete_path = os.path.join(data_path, '04_query_npz')
partial_path = os.path.join(data_path, '05_als_npz')

# complete_filelist = os.listdir(complete_path)
partial_filelist = os.listdir(partial_path)

trial_filelist = partial_filelist[:10]

datalist = []
num_faces = 2000

for i in range(len(trial_filelist)):
    part_data = np.load(os.path.join(partial_path, trial_filelist[i]))
    xyz = part_data['unit_als']
    hull_mesh = trimesh.convex.convex_hull(xyz)
    # vs, faces = m.vertices, m.faces
    mesh = remesh(hull_mesh, num_faces)