import os
import torch
import numpy as np
from torch import optim
from model import EncodeProcessDecode
from torch_geometric.data import DataLoader
from log import logger_setup
import time
from data_test import MeshAirfoilDataset,log_images
from utils import write_tecplot, write_tecplotzone
device = torch.device("cuda:0")
root_logger = logger_setup(os.path.join('/home/yangzhishuang/test2', 'logfioltest818.log'))
from torch_geometric.data import DataLoader 
model=EncodeProcessDecode(output_size=3,
                 latent_size=128,
                 num_layers=2,
                 message_passing_aggregator='sum', message_passing_steps=6).to(device)
criterion =torch.nn.MSELoss().to(device)                 
model.load_state_dict(torch.load('/home/yangzhishuang/test2/result/modelairfoil519.pkl'))                 
root_logger.info("===========start test===========") 
loader=MeshAirfoilDataset('/home/yangzhishuang/cfd-gcn/data/NACA0012_interpolate/',mode='test')
dataset_test=[]
for i in range(loader.len):
    data=loader.get(i)
    dataset_test.append(data)
test_loader = DataLoader(dataset_test, batch_size=4, shuffle=True)
model.eval()
sum_time=0
with torch.no_grad():
     sum_loss=0
     for batch in test_loader:
         batch=batch.to(device)
         #print(batch.num_graphs)
         truefield=batch.y
         start=time.perf_counter()
         prefield=model(batch)
         end=time.perf_counter()
         print((end-start)*1000)
         sum_time+=(end-start)
        # for i in range(batch.num_graphs):
        #   write_tecplotzone(batch.pos[i*6684:(i+1)*6684,:],prefield[i*6684:(i+1)*6684,:],
        #   loader.elems_list,loader.marker_dict,filename=f'airfoilwithbound-amgnet/aoa={batch.aoa[i]}mach={batch.mach_or_reynolds[i]}.dat')
        #   log_images(batch.pos, prefield,truefield, batch.batch,loader.elems_list, 'test',
       #     log_idx=i,file=f'resultfoil/aoa={batch.aoa[i]}mach={batch.mach_or_reynolds[i]}--')
         #plot_field(batch.pos,prefield,loader.elems_list)
         #write_tecplotzone(batch.pos,truefield,loader.elems_list)
         #log_images(batch.pos, prefield,truefield, batch.batch,loader.elems_list, 'test')
         #plot_field(batch.pos,prefield,loader.elems_list)
         mes_loss=criterion(prefield,truefield)
         loss=mes_loss.cpu()
         sum_loss+=loss.item()
sum_loss1=sum_loss/(len(test_loader))
print(sum_time/len(test_loader))
root_logger.info("        trajectory_loss")
root_logger.info("        " + str(sum_loss1))