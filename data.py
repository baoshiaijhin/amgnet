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


class MeshAirfoilDataset(Dataset):
    def __init__(self, root, mode='train'):
        self.mode = mode
        self.data_dir = Path(root) / ('outputs_' + mode)
        self.file_list = os.listdir(self.data_dir)
        self.len = len(self.file_list)

        self.mesh_graph = get_mesh_graph(Path(root) / 'mesh_fine.su2')

        # either [maxes, mins] or [means, stds] from data for normalization
        # with open(self.data_dir / 'train_mean_std.pkl', 'rb') as f:
        with open(self.data_dir.parent / 'train_max_min.pkl', 'rb') as f:
            self.normalization_factors = pickle.load(f)

        self.nodes = torch.from_numpy(self.mesh_graph[0])
        self.edges = torch.from_numpy(self.mesh_graph[1])
        self.elems_list = self.mesh_graph[2]
        self.marker_dict = self.mesh_graph[3]
        self.node_markers = self.nodes.new_full((self.nodes.shape[0], 1), fill_value=-1)
        for i, (marker_tag, marker_elems) in enumerate(self.marker_dict.items()):
            for elem in marker_elems:
                self.node_markers[elem[0]] = i
                self.node_markers[elem[1]] = i

        super().__init__(root)
        
    def __len__(self):
        return self.len

    def get(self, idx):
        with open(self.data_dir / self.file_list[idx], 'rb') as f:
            fields = pickle.load(f)
        fields = self.preprocess(fields)

        aoa, reynolds, mach = self.get_params_from_name(self.file_list[idx])
        aoa = aoa
        aoa = torch.from_numpy(aoa)
        mach_or_reynolds = mach if reynolds is None else reynolds
        mach_or_reynolds = torch.from_numpy(mach_or_reynolds)

        norm_aoa = aoa / 10
        norm_mach_or_reynolds = mach_or_reynolds if reynolds is None else (mach_or_reynolds - 1.5e6) / 1.5e6

        # add physics parameters to graph
        nodes = torch.cat([
            self.nodes,
            norm_aoa.unsqueeze(0).repeat(self.nodes.shape[0], 1),
            norm_mach_or_reynolds.unsqueeze(0).repeat(self.nodes.shape[0], 1),
            self.node_markers
        ], dim=-1)
        ######
        ######
        data = Data(x=nodes, y=fields, edge_index=self.edges,pos=self.nodes)
        ###########################
        sender=data.x[data.edge_index[0]]
        receiver=data.x[data.edge_index[1]]
        relation_pos=sender[:,0:2]-receiver[:,0:2]
        post=torch.norm(relation_pos,p=2,dim=1,keepdim=True)
        #data.edge_attr=torch.cat([relation_pos,post],dim=1)
        data.edge_attr=post
        #####归一化边特征
        std_epsilon=torch.tensor([1e-8])
        a=torch.mean(data.edge_attr,axis=0,dtype=torch.float32)
        b=data.edge_attr.std(dim=0)
        b=torch.maximum(b,std_epsilon)
        data.edge_attr=(data.edge_attr-a)/b
        ####归一化点特征
        #a=torch.mean(data.x,axis=0,dtype=torch.float32)
        #b=data.x.std(dim=0)
        #b=torch.maximum(b,std_epsilon)
        #data.x=(data.x-a)/b
        #########################
        data.aoa = aoa
        data.norm_aoa = norm_aoa
        data.mach_or_reynolds = mach_or_reynolds
        data.norm_mach_or_reynolds = norm_mach_or_reynolds
        return data

    def preprocess(self, tensor_list, stack_output=True):
        # data_means, data_stds = self.normalization_factors
        data_max, data_min = self.normalization_factors
        normalized_tensors = []
        for i in range(len(tensor_list)):
            # tensor_list[i] = (tensor_list[i] - data_means[i]) / data_stds[i] / 10
            normalized = (tensor_list[i] - data_min[i]) / (data_max[i] - data_min[i]) * 2 - 1
            if type(normalized) is np.ndarray:
                normalized = torch.from_numpy(normalized)
            normalized_tensors.append(normalized)
        if stack_output:
            normalized_tensors = torch.stack(normalized_tensors, dim=1)
        return normalized_tensors

    def _download(self):
        pass

    def _process(self):
        pass

    @staticmethod
    def get_params_from_name(filename):
        s = filename.rsplit('.', 1)[0].split('_')
        aoa = np.array(s[s.index('aoa') + 1])[np.newaxis].astype(np.float32)
        reynolds = s[s.index('re') + 1]
        reynolds = np.array(reynolds)[np.newaxis].astype(np.float32) if reynolds != 'None' else None
        mach = np.array(s[s.index('mach') + 1])[np.newaxis].astype(np.float32)
        return aoa, reynolds, mach

def  log_images(nodes, pred, true, batch, elems_list, mode, log_idx=0,iterate=0,file='field.png'):
        inds = batch == log_idx
        nodes = nodes[inds]
        pred = pred[inds] 
        true = true[inds]
        for field in range(pred.shape[1]):
            true_img = plot_field(nodes, elems_list, true[:, field],
                                  title='true')
            true_img = ToTensor()(true_img)
            min_max = (true[:, field].min().item(), true[:, field].max().item())

            pred_img = plot_field(nodes, elems_list, pred[:, field],
                                  title='pred',clim=min_max)  #clim=min_max 
            pred_img=ToTensor()(pred_img)
            imgs=[pred_img,true_img]  
            grid = make_grid(torch.stack(imgs), padding=0) 
            out_file=file+f'{field}'
            utils.save_image(grid,out_file+'_field.png')  

def plot_field(nodes, elems_list, field, contour=False, clim=None, zoom=True,
               get_array=True, out_file=None, show=False, title=''):
    elems_list = sum(elems_list, [])
    tris, _ = quad2tri(elems_list)
    tris = np.array(tris)
    x, y = nodes[:, :2].t().detach().cpu().numpy()
    field = field.detach().cpu().numpy()
    fig = plt.figure(dpi=800)
    if contour:
        plt.tricontourf(x, y, tris, field)
    else:
        plt.tripcolor(x, y, tris, field)
    if clim:
        plt.clim(*clim)
    plt.colorbar()
    if zoom:
        plt.xlim(left=-0.5, right=1.5)
        plt.ylim(bottom=-1.0, top=1.0)
    if title:
        plt.title(title)

    if out_file is not None:
        plt.savefig(out_file)
        plt.close()

    if show:
         plt.show()
        #raise NotImplementedError

    if get_array:
        fig.canvas.draw()
        a = np.fromstring(fig.canvas.tostring_rgb(),
                          dtype=np.uint8, sep='')
        a = a.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return a
def quad2tri(elems):
    new_elems = []
    new_edges = []
    for e in elems:
        if len(e) <= 3:
            new_elems.append(e)
        else:
            new_elems.append([e[0], e[1], e[2]])
            new_elems.append([e[0], e[2], e[3]])
            new_edges.append(torch.tensor(([[e[0]], [e[2]]]), dtype=torch.long))
    new_edges = torch.cat(new_edges, dim=1) if new_edges else torch.tensor([], dtype=torch.long)
    return new_elems, new_edges
