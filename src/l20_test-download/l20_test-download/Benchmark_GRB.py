import gurobipy
from gurobipy import GRB
import argparse
import random
import os
import numpy as np
import torch
from helper import get_a_new2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

#4 public datasets, IS, WA, CA, IP
TaskName='IP'
TestNum=100
def test_hyperparam(task):
    '''
    set the hyperparams
    k_0, k_1, delta
    '''
    if task=="IP":
        return 400,5,1
    elif task == "IS":
        return 300,300,15
    elif task == "WA":
        return 0,600,5
    elif task == "CA":
        return 400,0,10
    elif task == 'MIS':
        return 100, 20, 20
    elif task == 'MK':
        return 300, 20, 20
    else:
        return 400, 5, 10
k_0,k_1,delta=test_hyperparam(TaskName)

#set log folder
solver='GRB'
test_task = f'{TaskName}_{solver}_Benchmark'
if not os.path.isdir(f'./logs'):
    os.mkdir(f'./logs')
if not os.path.isdir(f'./logs/{TaskName}'):
    os.mkdir(f'./logs/{TaskName}')
if not os.path.isdir(f'./logs/{TaskName}/{test_task}'):
    os.mkdir(f'./logs/{TaskName}/{test_task}')
log_folder=f'./logs/{TaskName}/{test_task}'


#load pretrained model
if TaskName=="IP":
    #Add position embedding for IP model, due to the strong symmetry
    from GCN import GNNPolicy_position as GNNPolicy,postion_get
else:
    from GCN import GNNPolicy
model_name=f'{TaskName}.pth'
pathstr = f'./models/{model_name}'
#policy = GNNPolicy().to(DEVICE)
#state = torch.load(pathstr, map_location=torch.device('cuda:0'))
#policy.load_state_dict(state)


sample_names = sorted(os.listdir(f'./instance/test/{TaskName}'))
for ins_num in range(TestNum):
    test_ins_name = sample_names[ins_num]
    ins_name_to_read = f'./instance/test/{TaskName}/{test_ins_name}'

    #read instance
    gurobipy.setParam('LogToConsole', 1)  # hideout
    m = gurobipy.read(ins_name_to_read)
    m.Params.TimeLimit = 1000
    m.Params.Threads = 1
    m.Params.MIPFocus = 1
    m.Params.LogFile = f'{log_folder}/{test_ins_name}.log'

    # trust region method implemented by adding constraints
    instance_variabels = m.getVars()
    instance_variabels.sort(key=lambda v: v.VarName)
    variabels_map = {}
    for v in instance_variabels:  # get a dict (variable map), varname:var clasee
        variabels_map[v.VarName] = v
    m.optimize()






