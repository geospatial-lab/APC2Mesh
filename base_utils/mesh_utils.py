import numpy as np
import uuid
import trimesh
import torch, os
from typing import Tuple


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

#*************************** LOAD/SAVE FILES *****************************#
def save(file_name: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        f.write(trimesh.exchange.obj.export_obj(mesh))
        

def export(file, vs, faces, vn=None, color=None):
    with open(file, 'w+') as f:
        for vi, v in enumerate(vs):
            if color is None:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            else:
                f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
            if vn is not None:
                f.write("vn %f %f %f\n" % (vn[vi, 0], vn[vi, 1], vn[vi, 2]))
        for face in faces:
            f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))


def load(file_name: str):
    with open(file_name, "r") as f:
        mesh = trimesh.exchange.obj.load_obj(f)
    return np.float32(mesh["vertices"]), mesh["faces"]


def load_obj(file):
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces

#*************************** HELPER FUNCTIONS *****************************#
def populate_e(meshes, verts=None):
    mesh = meshes[0]
    if verts is None:
        verts = torch.rand(len(meshes), mesh.vs.shape[0], 3).to(mesh.vs.device)
    x = verts[:, mesh.edges, :]
    return x.view(len(meshes), mesh.edges_count, -1).permute(0, 2, 1).type(torch.float32)

def build_v(x, meshes):
    # mesh.edges[mesh.ve[2], mesh.vei[2]]
    mesh = meshes[0]  # b/c all meshes in batch are same
    x = x.reshape(len(meshes), 2, 3, -1)
    vs_to_sum = torch.zeros([len(meshes), len(mesh.vs_in), mesh.max_nvs, 3], dtype=x.dtype, device=x.device)
    x = x[:, mesh.vei, :, mesh.ve_in].transpose(0, 1)
    vs_to_sum[:, mesh.nvsi, mesh.nvsin, :] = x
    vs_sum = torch.sum(vs_to_sum, dim=2)
    nvs = mesh.nvs
    vs = vs_sum / nvs[None, :, None]
    return vs