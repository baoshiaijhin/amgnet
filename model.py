import collections
from math import ceil
from collections import OrderedDict
import functools
import torch
from torch import feature_alpha_dropout, nn as nn
import torch_scatter
from torch_scatter.composite import scatter_softmax
import torch.nn.functional as F
import os
from utils import batch_mm
import sys
# dirty hack: include top level folder to path
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)
from torch_geometric.nn import knn_interpolate
from gn_block import  Processor

device = torch.device('cuda:0')

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

class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, latent_size):
        super().__init__()
        self._make_mlp = make_mlp
        self._latent_size = latent_size
        self.node_model = self._make_mlp(latent_size)
        self.mesh_edge_model = self._make_mlp(latent_size)
        '''
        for _ in graph.edge_sets:
          edge_model = make_mlp(latent_size)
          self.edge_models.append(edge_model)
        '''

    def forward(self, graph):
        node_latents = self.node_model(graph.x)
        edge_latent = self.mesh_edge_model(graph.edge_attr)
        graph.x=node_latents
        graph.edge_attr=edge_latent
        return graph

class Decoder(nn.Module):
    """Decodes node features from graph."""
    # decoder = self._make_mlp(self._output_size, layer_norm=False)
    # return decoder(graph.node_features)

    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, output_size):
        super().__init__()
        self.model = make_mlp(output_size)

    def forward(self,node_features):
        return self.model(node_features)

class EncodeProcessDecode(nn.Module):
    """Encode-Process-Decode GraphNet model."""
    def __init__(self,
                 output_size,
                 latent_size,
                 num_layers,
                 message_passing_aggregator, message_passing_steps,
                 nodes
                 ):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self.min_nodes=nodes
        self._message_passing_steps = message_passing_steps
        self._message_passing_aggregator = message_passing_aggregator   
        self.encoder = Encoder(make_mlp=self._make_mlp, latent_size=self._latent_size)
        self.processor = Processor(make_mlp=self._make_mlp, output_size=self._latent_size,
                                   message_passing_steps=self._message_passing_steps,
                                   message_passing_aggregator=self._message_passing_aggregator,
                                   min_meshnodes=self.min_nodes,
                                   stochastic_message_passing_used=False)
        self.post_processor=self.makemlp(self._latent_size)                           
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               output_size=self._output_size)

    def _make_mlp(self, output_size, layer_norm=True):
        """Builds an MLP."""
        widths = [self._latent_size] * self._num_layers + [output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def spa_compute(self,x,p):
        j=len(x)-1
        node_features=x[j].x
        x_pos=x[j].pos
        for k in range(1,j+1):
            pos=p[-k]
            feature=knn_interpolate(node_features,pos,x[-(k+1)].pos)  
            node_features=x[-(k+1)].x+feature                  
            node_features=self.post_processor(node_features)
                    
        return node_features     
    def forward(self, graph):
        latent_graph = self.encoder(graph)
        x,p= self.processor(latent_graph)
        node_features=self.spa_compute(x,p)
        return self.decoder(node_features)