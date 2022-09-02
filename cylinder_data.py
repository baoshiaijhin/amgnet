import os
import pickle
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import math,copy
from os import PathLike
from typing import Sequence, Dict, Union, Tuple, List
import torch
from torch._six import string_classes
import collections.abc as container_abcs
from torch_geometric.data import Data, Batch, Dataset
from scipy.spatial.qhull import Delaunay
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision import utils
from utils import write_tecplot,write_data,write_tecplotzone
from torchvision.utils import make_grid
SU2_SHAPE_IDS = {
    'line': 3,
    'triangle': 5,
    'quad': 9,
}
def get_mesh_graph(mesh_filename: Union[str, PathLike],
                   dtype: np.dtype = np.float32
                   ) -> Tuple[np.ndarray, np.ndarray, List[List[List[int]]], Dict[str, List[List[int]]]]:
    def get_rhs(s: str) -> str:
        return s.split('=')[-1]

    marker_dict = {}
    with open(mesh_filename) as f:
        for line in f:
            if line.startswith('NPOIN'):
                num_points = int(get_rhs(line))
                mesh_points = [[float(p) for p in f.readline().split()[:2]]
                               for _ in range(num_points)]
                nodes = np.array(mesh_points, dtype=dtype)

            if line.startswith('NMARK'):
                num_markers = int(get_rhs(line))
                for _ in range(num_markers):
                    line = f.readline()
                    assert line.startswith('MARKER_TAG')
                    marker_tag = get_rhs(line).strip()
                    num_elems = int(get_rhs(f.readline()))
                    marker_elems = [[int(e) for e in f.readline().split()[-2:]]
                                    for _ in range(num_elems)]
                    # marker_dict[marker_tag] = np.array(marker_elems, dtype=np.long).transpose()
                    marker_dict[marker_tag] = marker_elems

            if line.startswith('NELEM'):
                edges = []
                triangles = []
                quads = []
                num_edges = int(get_rhs(line))
                for _ in range(num_edges):
                    elem = [int(p) for p in f.readline().split()]
                    if elem[0] == SU2_SHAPE_IDS['triangle']:
                        n = 3
                        triangles.append(elem[1:1+n])
                    elif elem[0] == SU2_SHAPE_IDS['quad']:
                        n = 4
                        quads.append(elem[1:1+n])
                    else:
                        raise NotImplementedError
                    elem = elem[1:1+n]
                    edges += [[elem[i], elem[(i+1) % n]] for i in range(n)]
                edges = np.array(edges, dtype=np.long).transpose()
                # triangles = np.array(triangles, dtype=np.long)
                # quads = np.array(quads, dtype=np.long)
                elems = [triangles, quads]


    return nodes, edges, elems, marker_dict

class MeshcylinderDataset(Dataset):
    def __init__(self, root, mode='train'):
        self.mode = mode
        self.data_dir = Path(root) / (mode)
        self.file_list = os.listdir(self.data_dir)
        self.len = len(self.file_list)

        self.mesh_graph = get_mesh_graph(Path(root) / 'cylinder.su2')

        # either [maxes, mins] or [means, stds] from data for normalization
        # with open(self.data_dir / 'train_mean_std.pkl', 'rb') as f:
        #with open(self.data_dir.parent / 'train_max_min.pkl', 'rb') as f:
        self.normalization_factors =torch.tensor([[978.6001,  48.9258,  24.8404],
        [-692.3159,   -6.9950,  -24.8572]])
        self.nodes = torch.from_numpy(self.mesh_graph[0])
        self.meshnodes=self.mesh_graph[0]
        self.edges = torch.from_numpy(self.mesh_graph[1])
        self.elems_list = self.mesh_graph[2]
        self.marker_dict = self.mesh_graph[3]
        self.bounder=[]
        self.node_markers = self.nodes.new_full((self.nodes.shape[0], 1), fill_value=-1)
        for i, (marker_tag, marker_elems) in enumerate(self.marker_dict.items()):
            for elem in marker_elems:
                self.node_markers[elem[0]] = i
                self.node_markers[elem[1]] = i

        super().__init__(root)
        
    def __len__(self):
        return self.len

    def get(self, idx):
        with open(self.data_dir/self.file_list[idx],'r') as f:
            field=[]
            pos=[]
            i=1
            for lines in f.readlines():
                if not i: 
                        lines=lines.rstrip('\n')
                        lines_pos=lines.split(',')[1:3]
                        lines_field=lines.split(',')[3:]
                        #for i in range(len(lines)):
                        #    a=float(lines[i])
                        numbers_float =list(eval(i) for i in lines_pos)
                        array=np.array(numbers_float,np.float32)
                        a=torch.from_numpy(array) 
                        pos.append(a)
                        numbers_float =list(eval(i) for i in lines_field)
                        array=np.array(numbers_float,np.float32)
                        a=torch.from_numpy(array) 
                        field.append(a)
                i=0
        field=torch.stack(field,axis=0)
        pos= torch.stack(pos,axis=0)       
        indexlist=[]
        f=open("2.txt","w")
        for i in range(self.meshnodes.shape[0]):
            b=torch.from_numpy(self.meshnodes[i:(i+1)])
            b=torch.squeeze(b)
            index=torch.nonzero(torch.sum((pos==b),dim=1,dtype=torch.float32)==pos.shape[1])
            f.write(f'{index}\n')
            indexlist.append(index) 
        f.close()    
        indexlist=torch.stack(indexlist,dim=0)
        indexlist=torch.squeeze(indexlist)
        fields=field[indexlist]
        #write_tecplotzone(self.nodes,fields,self.elems_list)
        velocity= self.get_params_from_name(self.file_list[idx])
        aoa = torch.from_numpy(velocity)

        #k=(1.0-0.5)/(40-1)
        #norm_aoa = 0.5+k*(aoa-1)
        norm_aoa=aoa/40
        # add physics parameters to graph
        nodes = torch.cat([
            self.nodes,
            norm_aoa.unsqueeze(0).repeat(self.nodes.shape[0], 1),
            self.node_markers
        ], dim=-1)
        ######
        ######
        data = Data(x=nodes, y=fields, edge_index=self.edges,pos=self.nodes,velocity=aoa)
        ###########################
        sender=data.x[data.edge_index[0]]
        receiver=data.x[data.edge_index[1]]
        relation_pos=sender[:,0:2]-receiver[:,0:2]
        post=torch.norm(relation_pos,p=2,dim=1,keepdim=True)
        data.edge_attr=post
        std_epsilon=torch.tensor([1e-8])
        a=torch.mean(data.edge_attr,axis=0,dtype=torch.float32)
        b=data.edge_attr.std(dim=0)
        b=torch.maximum(b,std_epsilon)
        data.edge_attr=(data.edge_attr-a)/b
        a=torch.mean(data.y,axis=0,dtype=torch.float32)
        b=data.y.std(dim=0)
        b=torch.maximum(b,std_epsilon)
        data.y=(data.y-a)/b
        data.norm_max = a
        data.norm_min = b
        
        "find the face of the boundery,our cylinder dataset come from fluent solver"
        with open('/home/fielddata/bounder','r') as f:
            field=[]
            pos=[]
            i=1
            for lines in f.readlines():
                if not i: 
                        lines=lines.rstrip('\n')
                        lines_pos=lines.split(',')[1:3]
                        lines_field=lines.split(',')[3:]
                        #for i in range(len(lines)):
                        #    a=float(lines[i])
                        numbers_float =list(eval(i) for i in lines_pos)
                        array=np.array(numbers_float,np.float32)
                        a=torch.from_numpy(array) 
                        pos.append(a)
                        numbers_float =list(eval(i) for i in lines_field)
                        array=np.array(numbers_float,np.float32)
                        a=torch.from_numpy(array) 
                        field.append(a)
                i=0
        field=torch.stack(field,axis=0)
        pos= torch.stack(pos,axis=0)       
        indexlist=[]
        for i in range(pos.shape[0]):
            b=pos[i:(i+1)]
            b=torch.squeeze(b)
            index=torch.nonzero(torch.sum((self.nodes==b),dim=1,dtype=torch.float32)==self.nodes.shape[1])
            indexlist.append(index) 
        indexlist=torch.stack(indexlist,dim=0)
        indexlist=torch.squeeze(indexlist)
        self.bounder=indexlist
        return data