import numpy as np
import trimesh

datapath = "sampling/data/noname1308907.obj"
data = np.array([[1164.098500, -1258.009500, -20.479850],
                 [1164.098500, -1258.009500, -23.374150],
                 [1168.120500, -1257.953500, -20.479850],
                 [1168.120500, -1257.953500, -23.374150],
                 [1163.911500, -1244.585500, -20.571850],
                 [1163.911500, -1244.585500, -23.374150],
                 [1167.933500, -1244.529500, -20.571850],
                 [1167.933500, -1244.529500, -23.374150]])

'''

def normalize(data, min_bound=-0.5, max_bound=0.5):
    xyzmin = np.min(data,axis=0)
    xyzmax = np.max(data, axis=0)

    ndata = (max_bound - min_bound) * (data - xyzmin) / (xyzmax - xyzmin) + min_bound
    return ndata, xyzmin, xyzmax

def denormalize(data, xyzmin, xyzmax, min_bound=-0.5, max_bound=0.5):
    odata = (data - min_bound) * (xyzmax - xyzmin) / (max_bound - min_bound) + xyzmin
    return odata

'''
def normalize(data):

    # Find the centroid
    print("Calculating centroid")
    centroid = data.mean(0)
    
    # subrtact centroid from the data
    recenteredData = data - centroid

    # Calculate Scale factor
    scale = np.abs(recenteredData).max()*2
    # Normalize
    normalized_data = np.divide(recenteredData,scale)

    return [normalized_data, scale, centroid]


def denormalize(data, scale, centroid):
   # un-normalize the data
   recenteredData = np.multiply(data,scale)

   translatedPoints = recenteredData + centroid

   return translatedPoints


def from_trimesh(file_in):
    mesh = trimesh.load(file_in)
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0/bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)
    return mesh

def plot_vertices(data):

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(data[:,0], data[:,1], data[:,2])
    plt.show()


[ndata, scale, centroid] = normalize(data)
plot_vertices(ndata)
nmesh = from_trimesh(file_in=datapath)
plot_vertices(nmesh.vertices)
odata = denormalize(ndata, scale, centroid)
plot_vertices(odata)

print(np.allclose(data,odata))