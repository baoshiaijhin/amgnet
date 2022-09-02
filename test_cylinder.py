import os
import torch
import numpy as np
from torch import optim
from model import EncodeProcessDecode
from torch_geometric.data import DataLoader
from tf_data import logger_setup
import time
from data_test import MeshAirfoilDataset,log_images
from utils import write_tecplot, write_tecplotzone,write_tecplotwallzone
device = torch.device("cuda:0")
from cylinder_data import MeshcylinderDataset
root_logger = logger_setup(os.path.join('/home/yangzhishuang/test2', 'logcylindertest826.log'))
from torch_geometric.data import DataLoader 
model=EncodeProcessDecode(output_size=3,
                 latent_size=128,
                 num_layers=2,
                 message_passing_aggregator='sum', message_passing_steps=6).to(device)
criterion =torch.nn.MSELoss().to(device)                 
#model.load_state_dict(torch.load('/home/yangzhishuang/test2/result/modelcylinderstd1.pkl'))
model.load_state_dict(torch.load('/home/yangzhishuang/test2/result822/modelcylinder-level-6-2000-822.pkl'))                   
root_logger.info("===========start test===========") 
loader=MeshcylinderDataset('/home/yangzhishuang/fielddata/cylinderdata',mode='test')
dataset_test=[]
for i in range(loader.len):
    data=loader.get(i)
    dataset_test.append(data)
test_loader = DataLoader(dataset_test, batch_size=4, shuffle=True)
model.eval()
with torch.no_grad():
     sum_loss=0
     for batch in test_loader:
         batch=batch.to(device)
         print(batch.velocity)
         truefield=batch.y
         #start=time.perf_counter()
         prefield=model(batch)
        # end=time.perf_counter()
         #print((end-start)*1000)
        # for i in range(batch.num_graphs):
        #  write_tecplotzone(batch.pos[i*3887:(i+1)*3887,:],prefield[i*3887:(i+1)*3887,:],
        #  loader.elems_list,filename=f'resultcylinderstd/{batch.velocity[i]}.dat')
        #  log_images(batch.pos, prefield,truefield, batch.batch,loader.elems_list, 'test',log_idx=i,
        #  file=f'resultcylinderstd/velocity={batch.velocity[i]}--')
         #plot_field(batch.pos,prefield,loader.elems_list)
       #  for i in range(batch.num_graphs):
        #  write_tecplotzone(batch.pos[i*3887:(i+1)*3887,:],prefield[i*3887:(i+1)*3887,:],
        #  loader.elems_list,filename=f'resultcylinderstd/{batch.velocity[i]}.dat')
        #   write_tecplotwallzone(batch.pos[i*3887:(i+1)*3887,:],prefield[i*3887:(i+1)*3887,:],loader.elems_list,loader.marker_dict,loader.bounder,
        #    filename=f'cylinderwithbound-amgnet/cylinder_velocity={batch.velocity[i]}.dat')
         mes_loss=criterion(prefield,truefield)
         loss=mes_loss.cpu()
         sum_loss+=loss.item()
sum_loss1=sum_loss/(len(test_loader))
root_logger.info("        trajectory_loss")
root_logger.info("        " + str(sum_loss1))