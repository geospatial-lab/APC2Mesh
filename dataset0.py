import os, random
import numpy as np
import torch
from torch.utils.data import dataloader
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset

def nonIdxSelect(list_var, idx):
    list_var.sort()
    list2arr = np.asarray(list_var)
    mask = np.ones(len(list_var), dtype=bool)
    mask[idx] = False
    return list2arr[mask], list2arr[~mask]

def createDatalist(train=True, npoints=4096, test_split_cnt=40):
        data_path = '/data/processed/%d' %(npoints)
        complete_path = os.path.join(data_path, '04_query_npz')
        partial_path = os.path.join(data_path, '05_als_npz')

        """train/test split"""
        complete_filelist = os.listdir(complete_path)
        partial_filelist = os.listdir(partial_path)

        assert len(complete_filelist) == len(partial_filelist), "partial and complete filelist count not the same, check!"

        test_idx = random.sample(range(0, len(complete_filelist)), test_split_cnt)

        complete_tr_list, complete_ts_list = nonIdxSelect(complete_filelist, test_idx)
        partial_tr_list, partial_ts_list = nonIdxSelect(partial_filelist, test_idx)

        datalist = []
        if train:
            for i in range(len(complete_tr_list)):
                comp_data = np.load(os.path.join(complete_path, complete_tr_list[i]))
                part_data = np.load(os.path.join(partial_path, partial_tr_list[i]))
                data = Data(cpos=torch.from_numpy(comp_data['query_pnts']), 
                            csdf=torch.from_numpy(comp_data['query_sdf']), 
                            ppos=torch.from_numpy(part_data['unit_als']), 
                            psdf=torch.from_numpy(part_data['unit_als_sdf']))
                datalist.append(data)
            return datalist

customDatalist = createDatalist(train=True, npoints=4096, test_split_cnt=40)
customLoader = DataLoader(customDatalist, batch_size=8)
print('Length of train dataset:', len(customLoader))

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

#     def nonIdxSelect(self, list_var, idx):
#         list_var.sort()
#         list2arr = np.asarray(list_var)
#         mask = np.ones(len(list_var), dtype=bool)
#         mask[idx] = False
#         return list2arr[mask], list2arr[~mask]

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

# BuildingDataset = OurDataset()
# tr_loader = DataLoader(BuildingDataset.datalist, batch_size=8, shuffle=True)
# print('Length of train dataset:', len(BuildingDataset))
from torch_geometric.nn import knn_graph
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for data in customLoader:
    # print('Number of points:', len(data.cpos))
    # print('shape of points:', data.cpos.shape)
    # print('sdf shape:', data.csdf.shape)
    # print('shape of als_points:', data.ppos.shape)
    # print('als_sdf shape:', data.psdf.shape)
    batch = torch.ones_like(data.csdf).to(device)
    edge_index = knn_graph(x=data.cpos.to(device), k=16, batch=batch, loop=True, flow='target_to_source')

# #     break
# pass

print('done ...')