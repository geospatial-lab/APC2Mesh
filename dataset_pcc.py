import os, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import trimesh
from mesh_to_sdf import get_surface_point_cloud
from pytictoc import TicToc
from point_ops.pointnet2_ops import pointnet2_utils

t = TicToc() #create instance of class

class CustomDataset(Dataset):

    def __init__(self, split='train', npoints=4096, test_split_cnt=40, device='cpu'):
        self.split = split
        self.device = device
        data_path = '/data/processed/%d' %(npoints)
        self.mesh_path = os.path.join(data_path, '03_snt_obj')
        self.partial_path = os.path.join(data_path, '05_als_npz')

        """train/test split"""
        mesh_filelist = os.listdir(self.mesh_path)
        partial_filelist = os.listdir(self.partial_path)

        assert len(mesh_filelist) == len(partial_filelist), "partial and mesh filelist count not the same, check!"

        test_idx = random.sample(range(0, len(mesh_filelist)), test_split_cnt)
        
        self.mesh_list = self.nonIdxSelect(mesh_filelist, test_idx)
        self.partial_list = self.nonIdxSelect(partial_filelist, test_idx)

    def __getitem__(self, index):
        partial_file = self.partial_list[index]
        mesh_file = self.mesh_list[index]
        partial_pc = np.load(os.path.join(self.partial_path, partial_file))['unit_als'][:,:]  # has normals
        mesh = trimesh.load(os.path.join(self.mesh_path, mesh_file))
        
        t.tic() #Start timer
        # complete_pc = trimesh.sample
        '''surf_instance contains various data, e.g. points, kd_tree, etc.'''
        surf_instance = get_surface_point_cloud(mesh, 
                                                surface_point_method='sample', 
                                                bounding_radius=1, 
                                                scan_count=30, 
                                                scan_resolution=200, 
                                                sample_point_count=50000, 
                                                calculate_normals=True)
        # t.toc('mesh sampling took') #Time elapsed since t.tic()

        # use fps to reduce points to fixed number
        surf_pnt_samples = torch.from_numpy(surf_instance.points).float()[None, :, :].to(self.device) # BN3
        surf_pnt_normals = torch.from_numpy(surf_instance.normals).float()[None, :, :].to(self.device) # BN3
        fps_idx = pointnet2_utils.furthest_point_sample(surf_pnt_samples, 16348) # xyz: torch.Tensor
        # t.toc('mesh/fps sampling took') #Time elapsed since t.tic()

        surf_pnt_samples = surf_pnt_samples.permute(0,2,1).contiguous()
        complete_pc = pointnet2_utils.gather_operation(surf_pnt_samples, fps_idx.int())
        complete_normals = pointnet2_utils.gather_operation(surf_pnt_normals.permute(0,2,1).contiguous(), fps_idx.int())
        # t.toc('mesh/fps sampling + indexing took') #Time elapsed since t.tic()
        complete_pc = torch.cat((torch.squeeze(complete_pc), torch.squeeze(complete_normals)),0).permute(1,0) # add .copy() to this line and change variable name if previous line has more use down the line

        return torch.from_numpy(partial_pc), complete_pc

    def __len__(self):
        return len(self.mesh_list)

    def nonIdxSelect(self, list_var, idx):
        list_var.sort()
        list2arr = np.asarray(list_var)
        mask = np.ones(len(list_var), dtype=bool)
        mask[idx] = False
        if self.split == 'train':
            return list2arr[mask]
        else: 
            return list2arr[~mask]

    def index_points(self, points, idx):
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

    def farthest_point_sample(self, xyz, npoint):
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


# class OurDataset(InMemoryDataset): # data.Dataset
#     def __init__(self, train=True, npoints=4096, test_split_cnt=40):
#         self.data_path = '/data/processed/%d' %(npoints)
#         self.complete_path = os.path.join(self.data_path, '04_query_npz')
#         self.partial_path = os.path.join(self.data_path, '05_als_npz')

#         self.npoints = npoints
#         self.train = train

#         """train/test split"""
#         complete_filelist = os.listdir(self.complete_path)
#         partial_filelist = os.listdir(self.partial_path)

#         assert len(complete_filelist) == len(partial_filelist), "partial and complete filelist count not the same, check!"

#         test_idx = random.sample(range(0, len(complete_filelist)), test_split_cnt)

#         # make files in both lists hold the same sequence

#         self.complete_tr_list, self.complete_ts_list = self.nonIdxSelect(complete_filelist, test_idx)
#         self.partial_tr_list, self.partial_ts_list = self.nonIdxSelect(partial_filelist, test_idx)

#     def process_filenames(self,):
#         datalist = []
#         if self.train:
            
#             for i in range(len(self.complete_tr_list)):
#                 comp_data = np.load(os.path.join(self.complete_path, self.complete_tr_list[i]))
#                 part_data = np.load(os.path.join(self.partial_path, self.partial_tr_list[i]))
#                 data = Data(cpos=torch.from_numpy(comp_data['query_pnts']), 
#                             csdf=torch.from_numpy(comp_data['query_sdf']), 
#                             ppos=torch.from_numpy(part_data['unit_als']), 
#                             psdf=torch.from_numpy(part_data['unit_als_sdf']))
#                 datalist.append(data)
#             return datalist

#     def process(self):
#         datalist = self.process_filenames()
#         torch.save(self.collate(datalist), self.datapath)

#         # if train:
#         #     # load train split
#         #     points = []
#         #     sdf = []
#         #     for file in complete_tr_list:
#         #         comp_data = np.load(os.path.join(self.complete_path, file))  
#         #         points.append(np.expand_dims(comp_data['query_pnts'], axis=0))
#         #         sdf.append(np.expand_dims(comp_data['query_sdf'], axis=0))

#         #     als_points = []
#         #     als_sdf = []
#         #     for file in partial_tr_list:
#         #         part_data = np.load(os.path.join(self.partial_path, file))  
#         #         als_points.append(np.expand_dims(part_data['unit_als'], axis=0))
#         #         als_sdf.append(np.expand_dims(part_data['unit_als_sdf'], axis=0))

#         #     self.points = np.concatenate(points, 0)
#         #     self.sdf = np.concatenate(sdf, 0)
#         #     self.als_points = np.concatenate(als_points, 0) # FIXME: issue of non-fixed npoints. resolve in sdf_try.py
#         #     self.als_sdf = np.concatenate(als_sdf, 0)
#         # else:
#         #     # load test split
#         #     for file in complete_ts_list:
#         #         comp_data = np.load(os.path.join(self.complete_path, file)) 
#         #         # self.points = comp_data['query_pnts']
#         #         # self.sdf = comp_data['query_sdf']

#         #     for file in partial_ts_list:
#         #         part_data = np.load(os.path.join(self.partial_path, file))  
#         #         # self.als_points = part_data['unit_als']
#         #         # self.als_sdf = part_data['unit_als_sdf']

#     def len(self):
#         if self.train:
#             return len(self.complete_tr_list)
#         else:
#            return self.als_points.shape[0]


#     # def __getitem__(self, index):
#     #     points = torch.from_numpy((self.points[index]))
#     #     sdf = torch.from_numpy((self.sdf[index]))
#     #     als_points = torch.from_numpy((self.als_points[index]))
#     #     als_sdf = torch.from_numpy((self.als_sdf[index]))
#     #     return points, sdf, als_points, als_sdf

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# BuildingDataset = CustomDataset(device=device)
# tr_loader = DataLoader(BuildingDataset, batch_size=8, shuffle=True)
# print('Length of train dataset:', len(BuildingDataset))

# for data in tr_loader:
#     print('shape of partial points:', data[0].shape) # [B 4096, 6] include normals
#     print('shape of complete points:', data[1].shape) # [B 16348, 3*] *for now, yet to add normals
#     # print('sdf shape:', data.csdf.shape)
#     # print('shape of als_points:', data.ppos.shape)
#     # print('als_sdf shape:', data.psdf.shape)

# # #     break
# # pass

# print('done ...')