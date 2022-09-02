import os
import sys
import torch
import numpy as np
from torch import optim,nn
from model import EncodeProcessDecode
from torch_geometric.data import DataLoader
from data import get_mesh_graph
import math
# dirty hack: include top level folder to path
from log import logger_setup
from data import MeshAirfoilDataset,log_images
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0")
summaryWriter=SummaryWriter("logs/log1")
dataset=[]
train_loader=MeshAirfoilDataset('/home/data/NACA0012_interpolate/',mode='train')
for i in range(train_loader.len):
    data=train_loader.get(i)
    dataset.append(data)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model=EncodeProcessDecode(output_size=3,
                 latent_size=128,
                 num_layers=2,
                 message_passing_aggregator='sum', message_passing_steps=6).to(device)
optimizers = optim.Adam(model.parameters(), lr=0.0005)#weight_decay=5e-4
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, 0.1 + 1e-6, last_epoch=-1)
criterion =torch.nn.MSELoss().to(device)
root_logger = logger_setup(os.path.join('/home/AMGNET/', 'logairfoil.log'))
model.train()
root_logger.info("===========start train===========") 
loss_history=[]
for epoch in range(500):
   sum_loss=0
   root_logger.info("Epoch"+str(epoch+1)) 
   for batch in loader:
      batch=batch.to(device)
      truefield=batch.y
      prefield=model(batch)
      #if epoch%9==0:
         #log_images(batch.pos, prefield,truefield, batch.batch,train_loader.elems_list, 'train')
      mes_loss=criterion(prefield,truefield)
      optimizers.zero_grad()
      mes_loss.backward()
      optimizers.step()
      loss=mes_loss
      sum_loss+=loss.item()
   loss_history.append(sum_loss/len(loader))
   sum_loss1=(sum_loss)/len(loader)
   summaryWriter.add_scalar("loss",math.sqrt(sum_loss1),epoch)
   root_logger.info("        trajectory_loss")
   root_logger.info("        " + str(sum_loss1))  
   if((epoch==60)|(epoch==100)|(epoch==140)|(epoch==170)):
      scheduler.step()   
torch.save(model.state_dict(),'/home/result/modelairfoil.pkl')   
##########################
# test
##########################
model=EncodeProcessDecode(output_size=3,
                 latent_size=128,
                 num_layers=2,
                 message_passing_aggregator='sum', message_passing_steps=6).to(device)
model.load_state_dict(torch.load('/home/yangzhishuang/test/result/model.pkl'))                 
root_logger.info("===========start test===========") 
test_loader=MeshAirfoilDataset('/home/data/NACA0012_machsplit_noshock/',mode='test')
dataset_test=[]
for i in range(test_loader.len):
    data=test_loader.get(i)
    dataset_test.append(data)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=True)
model.eval()
with torch.no_grad():
     sum_loss=0
     for batch in test_loader:
         batch=batch.to(device)
         truefield=batch.y
         prefield=model(batch)
         log_images(batch.pos, prefield,truefield, batch.batch,train_loader.elems_list, 'test')
         mes_loss=criterion(prefield,truefield)
         loss=mes_loss.cpu()
         loss=np.sqrt(loss)
         sum_loss+=loss.item()
sum_loss1=sum_loss/(len(test_loader))
root_logger.info("        trajectory_loss")
root_logger.info("        " + str(sum_loss1))

       
          
