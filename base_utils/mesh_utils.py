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

def face_areas_normals(faces, vs):
    face_normals = torch.cross(vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
                               vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :], dim=2)
    face_areas = torch.norm(face_normals, dim=2)
    face_normals = face_normals / face_areas[:, :, None]
    face_areas = 0.5*face_areas
    return face_areas, face_normals

def sample_surface(faces, vs, count):  # sample from a surf., like in trimesh.sample --> points & their normals
    """
    sample mesh surface
    sample method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
  
  Args
    ---------
    vs: vertices
    faces: triangle faces (torch.long)
    count: number of samples
    Return
    ---------
    samples: (count, 3) points in space on the surface of mesh
    normals: (count, 3) corresponding face normals for points
    """
    bsize, nvs, _ = vs.shape
    weights, normal = face_areas_normals(faces, vs)
    weights_sum = torch.sum(weights, dim=1)
    dist = torch.distributions.categorical.Categorical(probs=weights / weights_sum[:, None])
    face_index = dist.sample((count,))

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vs[:, faces[:, 0], :]
    tri_vectors = vs[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(1, 1, 2).reshape((bsize, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, count, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, count, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(count, 2, 1, device=vs.device, dtype=tri_vectors.dtype)

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    normals = torch.gather(normal, dim=1, index=face_index)

    return samples, normals

def mesh_area(mesh):
    vs = mesh.vs
    faces = mesh.faces
    v1 = vs[faces[:, 1]] - vs[faces[:, 0]]
    v2 = vs[faces[:, 2]] - vs[faces[:, 0]]
    area = torch.cross(v1, v2, dim=-1).norm(dim=-1)
    return area

def local_nonuniform_penalty(mesh):  #it has to do with the local variations in neighboring faces properties & penalizing that diff
    # non-uniform penalty
    area = mesh_area(mesh)
    diff = area[mesh.gfmm][:, 0:1] - area[mesh.gfmm][:, 1:]
    penalty = torch.norm(diff, dim=1, p=1)
    loss = penalty.sum() / penalty.numel()
    return loss
