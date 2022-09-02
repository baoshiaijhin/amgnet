import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from pyamg.classical.split import RS,CLJP
import numpy as np
from torch_sparse import SparseTensor, coalesce
from torch_sparse import transpose
from torch_sparse import spspmm
from scipy.spatial.qhull import Delaunay
from torch.nn import LayerNorm
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
device=torch.device("cuda:0")
def getcorsenode(latent_graph):
    A=to_scipy_sparse_matrix(latent_graph.edge_index).tocsr()
    #splitting=RS(A)
    splitting=RS(A)
    index=np.array(np.nonzero(splitting))
    b = torch.from_numpy(index)
    b=torch.squeeze(b)
    return b

def StAS(index_A, value_A, index_S, value_S,N, kN,nor):
    r"""come from Ranjan, E., Sanyal, S., Talukdar, P. (2020, April). Asap: Adaptive structure aware pooling
        for learning hierarchical graph representations. AAAI(2020)"""

    index_A, value_A = coalesce(index_A, value_A, m=N, n=N)
    index_S, value_S = coalesce(index_S, value_S, m=N, n=kN)
    index_B, value_B = spspmm(index_A, value_A, index_S, value_S, N, N, kN)
    index_St, value_St = transpose(index_S, value_S, N, kN)
    index_B, value_B = coalesce(index_B, value_B, m=N, n=kN)                             
    # index_E, value_E = spspmm(index_St.cpu(), value_St.cpu(), index_B.cpu(), value_B.cpu(), kN, N, kN)
    index_E, value_E = spspmm(index_St, value_St, index_B, value_B, kN, N, kN)

    # return index_E.to(device), value_E.to(device)
    return index_E, value_E


def graph_connectivity(perm, edge_index, edge_weight, score,pos,N,nor):
    r"""come from Ranjan, E., Sanyal, S., Talukdar, P. (2020, April). Asap: Adaptive structure aware pooling
        for learning hierarchical graph representations. AAAI(2020)"""""
    
    kN = perm.size(0)
    perm2 = perm.view(-1, 1)
    
    # mask contains bool mask of edges which originate from perm (selected) nodes
    mask = (edge_index[0]==perm2).sum(0, dtype=torch.bool)
    
    # create the S
    S0 = edge_index[1][mask].view(1, -1)
    S1 = edge_index[0][mask].view(1, -1)
    index_S = torch.cat([S0, S1], dim=0)
    value_S = score[mask].detach().squeeze()
    
    # relabel for pooling ie: make S [N x kN]
    n_idx = torch.zeros(N, dtype=torch.long)
    n_idx[perm] = torch.arange(perm.size(0))
    index_S[1] = n_idx[index_S[1]]
    ##
    subgraphnode_pos=pos[perm].cpu()
    subgraphnode_pos=subgraphnode_pos.to(device)
    # create A
    index_A = edge_index.clone()
    if edge_weight is None:
        value_A = value_S.new_ones(edge_index[0].size(0))
    else:
        value_A = edge_weight.clone()
    value_A=torch.squeeze(value_A)
    #fill_value=1.0
    attrlist=[]
    for i in range(128):
        index_E, value_E = StAS(index_A, value_A[:,i], index_S, value_S, N, kN,nor)
        index_E, value_E = remove_self_loops(edge_index=index_E, edge_attr=value_E)
        #index_E, value_E = add_remaining_self_loops(edge_index=index_E, edge_attr=value_E, 
        #    fill_value=fill_value, num_nodes=kN)
        attrlist.append(value_E)
    edge_weight=torch.stack(attrlist,dim=1)      
    
    return index_E,edge_weight,subgraphnode_pos


def write_tecplotzone(pos,fields,elems_list,bounder_dict,filename='flowtest2.dat'):
    x = pos
    quad=elems_list[1]
    number=len(quad)
    quadlist=[]
    for res in quad:
         quadlist.extend(res)
    newquad=list(set(quadlist))
    x=x[newquad]
    afields=fields[newquad]
    num_nodes=x.size(0)
    N=len(quadlist)     
    n_idx = torch.zeros(N, dtype=torch.long)
    n_idx[newquad] = torch.arange(len(newquad))
    newquad = n_idx[quadlist]
    i=0
    elemlist=[]
    while(i<len(newquad)):
      a=[]
      a.append(int(newquad[i]))
      a.append(int(newquad[i+1]))
      a.append(int(newquad[i+2]))
      a.append(int(newquad[i+3]))
      elemlist.append(a)
      i+=4
    with open(filename, 'w') as f:
        f.write('TITLE = "Visualization of the volumetric solution"\n')
        f.write('VARIABLES = "CoordinateX"\n"CoordinateY"\n"CoordinateZ"\n"Component Velocity"\n"Pressure"\n"Component Velocity"\n')
        f.write('ZONE T="unspecified Step 1 Incr 0"\n')
        f.write(' STRANDID=1, SOLUTIONTIME=0\n')
        f.write(f' Nodes={num_nodes}, Elements={number}, ''ZONETYPE=FEQUADRILATERAL\n')
        f.write(' DATAPACKING=POINT\n')       
        f.write(' AUXDATA Time="0.000000e+00"\n')
        f.write(' DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE )\n')       
        for node, field in zip(x, afields):
            f.write(f'{node[0].item()}\t{node[1].item()}\t0.0\t'
                    f'{field[0].item()}\t{field[1].item()}\t'
                    f'{field[2].item()}\n')

        for elem in elemlist:
            f.write('\t'.join(str(x+1) for x in elem))
            if len(elem) == 3:
                # repeat last vertex if triangle
                f.write(f'\t{elem[-1]+1}')
            f.write('\n')
        quad=elems_list[0]
        number=len(quad)
        quadlist=[]
        for res in quad:
            quadlist.extend(res)
        newquad=list(set(quadlist))
        x=pos[newquad]
        bfields=fields[newquad]
        num_nodes=x.size(0)
        N=len(quadlist)     
        n_idx = torch.zeros(10000, dtype=torch.long)
        n_idx[newquad] = torch.arange(len(newquad))
        newquad = n_idx[quadlist]
        i=0
        elemlist=[]
        while(i<len(newquad)):
            a=[]
            a.append(int(newquad[i]))
            a.append(int(newquad[i+1]))
            a.append(int(newquad[i+2]))
            elemlist.append(a)
            i+=3
        f.write('ZONE T="unspecified Step 1 Incr 0"\n')
        f.write(' STRANDID=1, SOLUTIONTIME=0\n')
        f.write(f' Nodes={num_nodes}, Elements={number}, ''ZONETYPE=FETRIANGLE\n')
        f.write(' DATAPACKING=POINT\n') 
        f.write('AUXDATA Time="0.000000e+00"\n')
        f.write('DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE )\n') 
        for node, field in zip(x, bfields):
            f.write(f'{node[0].item()}\t{node[1].item()}\t0.0\t'
                    f'{field[0].item()}\t{field[1].item()}\t'
                    f'{field[2].item()}\n')

        for elem in elemlist:
            f.write('\t'.join(str(x+1) for x in elem))
            #if len(elem) == 3:
                # repeat last vertex if triangle
            #    f.write(f'\t{elem[-1]+1}')
            f.write('\n')        
        bound=bounder_dict['airfoil']
        bound_node_list=[]
        for node in bound:
            bound_node_list.extend(node)
        newnode=list(set(bound_node_list))
        x=pos[newnode]
        bound_field=fields[newnode] 
        Element_number=len(bound)
        num_nodes=x.size(0);     
        N=len(bound_node_list)     
        n_idx = torch.zeros(10000, dtype=torch.long)
        n_idx[newnode] = torch.arange(len(newnode))
        newnode = n_idx[bound_node_list]
        i=0
        elemlist=[]
        while(i<len(newnode)):
            a=[]
            a.append(int(newnode[i]))
            a.append(int(newnode[i+1]))
            elemlist.append(a)
            i+=2       
        f.write('ZONE T="Bounder"\n')
        f.write(' STRANDID=2, SOLUTIONTIME=0\n')
        f.write(f' Nodes={num_nodes}, Elements={Element_number}, ''ZONETYPE=FELineSeg\n')
        f.write(' DATAPACKING=POINT\n') 
        f.write('AUXDATA Common.BoundaryCondition="Wall"\n')
        f.write('AUXDATA Common.IsBoundaryZone="TRUE"\n')
        f.write('DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE )\n') 
        for node, field in zip(x, bound_field):
            f.write(f'{node[0].item()}\t{node[1].item()}\t0.0\t'
                    f'{field[0].item()}\t{field[1].item()}\t'
                    f'{field[2].item()}\n') 
        for elem in elemlist:
            f.write('\t'.join(str(x+1) for x in elem))
            #if len(elem) == 3:
                # repeat last vertex if triangle
            #    f.write(f'\t{elem[-1]+1}')
            f.write('\n')            

def write_tecplotwallzone(pos,fields,elems_list,makert,wall,filename='flowcfdgcn.dat'):
    
    unelement=makert
    x = pos
    quad=elems_list[1]
    number=len(quad)
    quadlist=[]
    for res in quad:
         quadlist.extend(res)
    newquad=list(set(quadlist))
    x=x[newquad]
    afields=fields[newquad]
    num_nodes=x.size(0)
    N=len(quadlist)     
    n_idx = torch.zeros(N, dtype=torch.long)
    n_idx[newquad] = torch.arange(len(newquad))
    newquad = n_idx[quadlist]
    i=0
    elemlist=[]
    while(i<len(newquad)):
      a=[]
      a.append(int(newquad[i]))
      a.append(int(newquad[i+1]))
      a.append(int(newquad[i+2]))
      a.append(int(newquad[i+3]))
      elemlist.append(a)
      i+=4
    with open(filename, 'w') as f:
        f.write('TITLE = "Visualization of the volumetric solution"\n')
        f.write('VARIABLES = "CoordinateX"\n"CoordinateY"\n"CoordinateZ"\n"Pressure"\n"Component Velocity"\n"Component Velocity"\n')
        f.write('ZONE T="unspecified Step 1 Incr 0"\n')
        f.write(' STRANDID=1, SOLUTIONTIME=0\n')
        f.write(f' Nodes={num_nodes}, Elements={number}, ''ZONETYPE=FEQUADRILATERAL\n')
        f.write(' DATAPACKING=POINT\n')       
        f.write(' AUXDATA Time="0.000000e+00"\n')
        f.write(' DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE )\n')       
        for node, field in zip(x, afields):
            f.write(f'{node[0].item()}\t{node[1].item()}\t0.0\t'
                    f'{field[0].item()}\t{field[1].item()}\t'
                    f'{field[2].item()}\n')

        for elem in elemlist:
            f.write('\t'.join(str(x+1) for x in elem))
            if len(elem) == 3:
                # repeat last vertex if triangle
                f.write(f'\t{elem[-1]+1}')
            f.write('\n')
        quad=elems_list[0]
        number=len(quad)
        quadlist=[]
        for res in quad:
            quadlist.extend(res)
        newquad=list(set(quadlist))
        x=pos[newquad]
        bfields=fields[newquad]
        num_nodes=x.size(0)
        N=len(quadlist)     
        n_idx = torch.zeros(10000, dtype=torch.long)
        n_idx[newquad] = torch.arange(len(newquad))
        newquad = n_idx[quadlist]
        i=0
        elemlist=[]
        while(i<len(newquad)):
            a=[]
            a.append(int(newquad[i]))
            a.append(int(newquad[i+1]))
            a.append(int(newquad[i+2]))
            elemlist.append(a)
            i+=3
        f.write('ZONE T="unspecified Step 1 Incr 0"\n')
        f.write(' STRANDID=1, SOLUTIONTIME=0\n')
        f.write(f' Nodes={num_nodes}, Elements={number}, ''ZONETYPE=FETRIANGLE\n')
        f.write(' DATAPACKING=POINT\n') 
        f.write('AUXDATA Time="0.000000e+00"\n')
        f.write('DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE )\n') 
        for node, field in zip(x, bfields):
            f.write(f'{node[0].item()}\t{node[1].item()}\t0.0\t'
                    f'{field[0].item()}\t{field[1].item()}\t'
                    f'{field[2].item()}\n')

        for elem in elemlist:
            f.write('\t'.join(str(x+1) for x in elem))
            #if len(elem) == 3:
                # repeat last vertex if triangle
            #    f.write(f'\t{elem[-1]+1}')
            f.write('\n')               
       ####
       ####
        newnode=wall
        x=pos[newnode]
        bound_field=fields[newnode]  
        num_nodes=x.size(0);     
        elemlist=[]
        point=[]
        for i in range(len(unelement['Unspecified'])):
         point.append(pos[unelement['Unspecified'][i][0]])
        point=torch.stack(point,axis=0) 
        for i in range(x.shape[0]):
            a=[]
            b=x[i]
            b=torch.squeeze(b)
            index=torch.nonzero(torch.sum((point==b),dim=1,dtype=torch.float32)==point.shape[1])
            c=unelement['Unspecified'][int(torch.squeeze(index))][1]
            k=pos[c]
            second=torch.nonzero(torch.sum((x==k),dim=1,dtype=torch.float32)==x.shape[1])
            a.append(i)
            a.append(int(torch.squeeze(second)))    
            elemlist.append(a)  
        f.write('ZONE T="Bounder"\n')
        f.write(' STRANDID=2, SOLUTIONTIME=0\n')
        f.write(f' Nodes={num_nodes}, Elements={80}, ''ZONETYPE=FELineSeg\n')
        f.write(' DATAPACKING=POINT\n') 
        f.write('AUXDATA Common.BoundaryCondition="Wall"\n')
        f.write('AUXDATA Common.IsBoundaryZone="TRUE"\n')
        f.write('DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE )\n') 
        for node, field in zip(x, bound_field):
            f.write(f'{node[0].item()}\t{node[1].item()}\t0.0\t'
                    f'{field[0].item()}\t{field[1].item()}\t'
                    f'{field[2].item()}\n') 
       
        for elem in elemlist:
            f.write('\t'.join(str(x+1) for x in elem))
            #if len(elem) == 3:
                # repeat last vertex if triangle
            #    f.write(f'\t{elem[-1]+1}')
            f.write('\n')                    