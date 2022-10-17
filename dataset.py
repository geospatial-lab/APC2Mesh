import os, random
import numpy as np
import torch
from torch.utils import data

class OurDataset(data.Dataset):
    def __init__(self, train=True, npoints=4096, test_split_cnt=40):
        self.data_path = '/data/processed/%d' %(npoints)
        self.complete_path = os.path.join(self.data_path, '04_query_npz')
        self.partial_path = os.path.join(self.data_path, '05_als_npz')

        """train/test split"""
        complete_filelist = os.listdir(self.complete_path)
        partial_filelist = os.listdir(self.partial_path)

        assert len(complete_filelist) == len(partial_filelist), "partial and complete filelist count not the same, check!"

        test_idx = random.sample(range(0, len(complete_filelist)), test_split_cnt)

        # make files in both lists hold the same sequence

        complete_tr_list, complete_ts_list = self.nonIdxSelect(complete_filelist, test_idx)
        partial_tr_list, partial_ts_list = self.nonIdxSelect(partial_filelist, test_idx)

        if train:
            # load train split
            points = []
            sdf = []
            for file in complete_tr_list:
                comp_data = np.load(os.path.join(self.complete_path, file))  
                points.append(np.expand_dims(comp_data['query_pnts'], axis=0))
                sdf.append(np.expand_dims(comp_data['query_sdf'], axis=0))

            als_points = []
            als_sdf = []
            for file in partial_tr_list:
                part_data = np.load(os.path.join(self.partial_path, file))  
                als_points.append(np.expand_dims(part_data['unit_als'], axis=0))
                als_sdf.append(np.expand_dims(part_data['unit_als_sdf'], axis=0))

            self.points = np.concatenate(points, 0)
            self.sdf = np.concatenate(sdf, 0)
            # self.als_points = np.concatenate(als_points, 0) # issue of non-fixed npoints. resolve in sdf_try.py
            # self.als_sdf = np.concatenate(als_sdf, 0)
        else:
            # load test split
            for file in complete_ts_list:
                comp_data = np.load(os.path.join(self.complete_path, file)) 
                # self.points = comp_data['query_pnts']
                # self.sdf = comp_data['query_sdf']

            for file in partial_ts_list:
                part_data = np.load(os.path.join(self.partial_path, file))  
                # self.als_points = part_data['unit_als']
                # self.als_sdf = part_data['unit_als_sdf']

        self.npoints = npoints
        self.train = train

    def nonIdxSelect(self, list_var, idx):
        list_var.sort()
        list2arr = np.asarray(list_var)
        mask = np.ones(len(list_var), dtype=bool)
        mask[idx] = False
        return list2arr[mask], list2arr[~mask]

    def __len__(self):
        if self.train:
            return self.points.shape[0]
        else:
           return self.als_points.shape[0]

    def __getitem__(self, index):
        points = torch.from_numpy((self.points[index]))
        sdf = torch.from_numpy((self.sdf[index]))
        # als_points = torch.from_numpy((self.als_points[index]))
        # als_sdf = torch.from_numpy((self.als_sdf[index]))
        return points, sdf #, als_points, als_sdf

BuildingDataset = OurDataset()
tr_loader = data.DataLoader(BuildingDataset, batch_size=8, shuffle=True)
print('Length of train dataset:', len(BuildingDataset))

for data in tr_loader:
    print('Number of points:', len(data))
    print('shape of points:', data[0].shape)
    print('sdf shape:', data[1].shape)
    # print('shape of als_points:', data[2].shape)
    # print('als_sdf shape:', data[3].shape)

#     break
pass

print('done ...')