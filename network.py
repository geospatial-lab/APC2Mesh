from dataset_pcc import OurDataset
from torch.utils import data
from torch import nn
import torch
# import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from deltaconv.nn import DeltaConv


class Net(nn.Module):
    def __init__(self, in_channels, conv_channels, mlps, num_nbrs, grad_reg, grad_kernel_width, centralize_first=True):
        """
        conv_channels params: (List[int]) channel output for each convolution
        num_nbrs params: (int) number of neighbors to use in estimating gradient
        grad_reg params: (float) the regularization value (lambda) used in least squares fitting 
        grad_kernel_width params: (float) the width of the gaussian kernel used to weight the least-squares problem 
                                  to approximate the gradient.
        centralize_first params: (bool) whether or not to centralize the input features
        """
        super().__init__()
        self.k = num_nbrs
        self.grad_reg = grad_reg
        self.grad_kernel_width = grad_kernel_width
        
        conv_channels.insert(0, in_channels)
        self.convs = nn.ModuleList()
        for i in range(len(conv_channels) - 1):
            last_layer = i == (len(conv_channels) - 2)
            self.convs.append(DeltaConv(conv_channels[i], 
                                        conv_channels[i+1], 
                                        depth=mlps, 
                                        centralized=(centralize_first and i == 0), # centralizes on the first layer
                                        vector=not(last_layer) # vector is only false if i is the last layer
                                        )
                                )

    def forward(self, x):
        points = x[0]
        als_ppoints = x[2]
        pos = torch.cat([points,als_ppoints], axis=1)

        """Operator construction"""
        # Create a kNN graph, which is used to:
        # 1) Perform maximum aggregation in the scalar stream.
        # 2) Approximate the gradient and divergence oeprators
        # TODO: try to incorporate attention-based aggregation instead of max or mean.
        batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        edge_index = knn_graph(pos, self.k, batch=batch, loop=True, flow='target_to_source')

        # Use the normals provided by the data or estimate a normal from the data.
        #   It is advised to estimate normals as a pre-transform.
        # TODO: add normal estimation to the dataset prep step so that we save time and "build_tangent_basis" directly
        edge_index_normal = knn_graph(pos, 10, batch=None, loop=True, flow='target_to_source')
        # When normal orientation is unknown, we opt for a locally consistent orientation.
        normal, x_basis, y_basis = estimate_basis(pos, edge_index_normal, orientation=pos)

                # Build the gradient and divergence operators.
        # grad and div are two sparse matrices in the form of SparseTensor.
        grad, div = build_grad_div(pos, normal, x_basis, y_basis, edge_index, batch=None, 
                                   kernel_width=self.grad_kernel_width, regularizer=self.grad_regularizer)
        
        """Forward pass convolutions"""
        # The scalar features are stored in x
        # x = data.x if hasattr(data, 'x') and data.x is not None else pos
        x = pos
        # TODO: explore data.x from the original data to figure out what exactly is in data.x
        # Vector features in v
        v = grad @ x # for now x is replaced with pos as stipulated from 3 lines up 
        
        # Store each of the interim outputs in a list
        out = []
        for conv in self.convs:
            x, v = conv(x, v, grad, div, edge_index)
            out.append(x)
        print('done !!')
        # TODO: Add the mlp part from DeltaNet** in accordance with encoder output for pnt_completion
        # TODO: This encoder later to be modified for edge and corner output
        return out

        
BuildingDataset = OurDataset(train=True, npoints=2048, test_split_cnt=40)
tr_loader = data.DataLoader(BuildingDataset, batch_size=8, shuffle=True)
model = Net(3, [64,128,256], 2, 16, 0.003, 1, True)
# Determine the device to run the experiment on.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for i, data in enumerate(tr_loader):
    data[0] = data[0].to(device)
    data[2] = data[2].to(device)
    output = model(data)

    print('done !!')