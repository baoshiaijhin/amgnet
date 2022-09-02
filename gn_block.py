from turtle import shapesize
import torch.nn as nn
import torch
import os
import sys
import collections
import torch_scatter
import scipy.sparse as sp
from math import ceil
from collections import OrderedDict
from torch_geometric.utils import to_dense_adj,subgraph,to_scipy_sparse_matrix,from_scipy_sparse_matrix
from pyamg.classical.split import RS
from utils import getcorsenode,graph_connectivity
from torch.nn import LayerNorm
import numpy as np
from scipy.spatial.qhull import Delaunay
from torch_sparse import spspmm
from torch_geometric.data import Data, Batch, Dataset
# dirty hack: include top level folder to path
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)
from tf_data import generate_adj
device=torch.device('cuda:0')
min_nodes=2000  #Each mesh can be coarsened to have no fewer points than this value
class Processor(nn.Module):
    '''
    This class takes the nodes with the most influential feature (sum of square)
    The the chosen numbers of nodes in each ripple will establish connection(features and distances) with the most influential nodes and this connection will be learned
    Then the result is add to output latent graph of encoder and the modified latent graph will be feed into original processor

    Option: choose whether to normalize the high rank node connection
    '''

    def __init__(self, make_mlp, output_size, message_passing_steps, message_passing_aggregator,attention=False,
                 stochastic_message_passing_used=False):
        super().__init__()
        self.stochastic_message_passing_used = stochastic_message_passing_used
        self.graphnet_blocks = nn.ModuleList()
        self.cofe_edge_blocks=nn.ModuleList()
        self.pool_blocks =nn.ModuleList()
        self.latent_size=output_size
        self.normalization=LayerNorm(128)
        for index in range(message_passing_steps):
            self.graphnet_blocks.append(GraphNetBlock(model_fn=make_mlp, output_size=output_size,
                                                      message_passing_aggregator=message_passing_aggregator,
                                                      attention=attention))
            #self.pool_blocks.append(ASAP_Pooling(in_channels=output_size,ratio=0.8,mlpfn=make_mlp))
            self.pool_blocks.append(GraphNetBlock(model_fn=make_mlp, output_size=output_size,
                                                      message_passing_aggregator=message_passing_aggregator,
                                                      attention=attention))                                         
    def forward(self, latent_graph, normalized_adj_mat=None):
            x=[]
            pos=[]
            new=[]
            for (graphnet_block,pool) in zip(self.graphnet_blocks,self.pool_blocks):
                if latent_graph.x.shape[0]>min_nodes:  
                    pre_matrix=graphnet_block(latent_graph)
                    x.append(pre_matrix)
                    cofe_graph=pool(pre_matrix)  #updata edge features
                    coarsenodes=getcorsenode(pre_matrix).to(device)
                    nodesfeatures=cofe_graph.x[coarsenodes]
                    subedge_index, edge_weight,subpos=graph_connectivity(perm=coarsenodes,
                    edge_index=cofe_graph.edge_index,
                    edge_weight=cofe_graph.edge_attr,score=cofe_graph.edge_attr[:,0]
                    ,pos=cofe_graph.pos,N=cofe_graph.x.size(0),nor=self.normalization)  
                    edge_weight=self.normalization(edge_weight)
                    pos.append(subpos)
                    latent_graph=Data(x=nodesfeatures,pos=subpos,
                    edge_attr=edge_weight,edge_index=subedge_index)
                else:
                      latent_graph=graphnet_block(latent_graph)
                      new.append(latent_graph)
            if len(new):
                x.append(new[-1])           
            return x,pos
  ####################################################################################################################
  #   GN block struct come from Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W. Battaglia. Learning
  #   mesh-based simulation with graph networks. In International Conference on Learning Representations, 2021.
  ####################################################################################################################
class GraphNetBlock(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, output_size, message_passing_aggregator, attention=False):
        super().__init__()
        self.edge_model = model_fn(output_size)
        self.node_model = model_fn(output_size)
        #if attention:
            #self.attention_model = AttentionModel()
        self.message_passing_aggregator = message_passing_aggregator

    def _update_edge_features(self,graph):
        """Aggregrates node features, and applies edge function."""
        senders =graph.edge_index[0].to(device)
        receivers = graph.edge_index[1].to(device)
        sender_features = torch.index_select(input=graph.x, dim=0, index=senders)
        receiver_features = torch.index_select(input=graph.x, dim=0, index=receivers)
        features = [sender_features, receiver_features,graph.edge_attr]
        features = torch.cat(features, dim=-1)
        return self.edge_model(features)

    '''
    def _update_node_features_mp_helper(self, features, receivers, add_intermediate):
        for index, feature_tensor in enumerate(features):
            des_index = receivers[index]
            add_intermediate[des_index].add(feature_tensor)
        return add_intermediate
    '''

    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape)
        if operation == 'sum':
            result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(data.float(), segment_ids, dim=0, dim_size=num_segments)
        result = result.type(data.dtype)
        return result

    def _update_node_features(self, node_features, edge_attr,edge_index):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[0]
        features = [node_features]
        features.append(
                    self.unsorted_segment_operation(edge_attr,edge_index[1], num_nodes,
                                                    operation=self.message_passing_aggregator))
        features = torch.cat(features, dim=-1)
        return self.node_model(features)

    def forward(self, graph):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions

        new_edge_features = self._update_edge_features(graph)
        # apply node function
        new_node_features = self._update_node_features(graph.x,graph.edge_attr,graph.edge_index)

        # add residual connections
        new_node_features += graph.x
        new_edge_features+=graph.edge_attr 
        return Data(x=new_node_features,pos=graph.pos,edge_attr=graph.edge_attr,
        edge_index=graph.edge_index,batch=graph.batch)
class LazyMLP(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input):
        input = input.to(device)
        y = self.layers(input)
        return y                   
def make_mlpnet(output_size, layer_norm=True):
        """Builds an MLP."""
        widths = [128] * 1 + [output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network        


