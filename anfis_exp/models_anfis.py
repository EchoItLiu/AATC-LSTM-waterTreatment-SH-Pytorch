import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import math
from torch.nn.utils import weight_norm
import numpy as np

 
"""ANFIS for fixing premise parameters"""    
class ANFIS_premise(nn.Module):
 
    def __init__(self, input_size = 9, ouput_size = 1):
        super(ANFIS, self).__init__()
        
        """ set all premise parameters """        
        #F1 generalized bell function 1
        self.a1 = nn.Parameter(torch.from_numpy(np.array([2.5])).cuda().float())
        self.b1 = nn.Parameter(torch.from_numpy(np.array([0.6])).cuda().float())
        self.c1 = nn.Parameter(torch.from_numpy(np.array([0.8])).cuda().float())
                    
        #F2  generalized bell function 2
        self.a2 = nn.Parameter(torch.from_numpy(np.array([2])).cuda().float())
        self.b2 = nn.Parameter(torch.from_numpy(np.array([0.5])).cuda().float())        
        self.c2 = nn.Parameter(torch.from_numpy(np.array([1])).cuda().float()) 
        
        #F3  Gaussian function 1
        self.a3 = nn.Parameter(torch.from_numpy(np.array([2])).cuda().float())
        self.c3 = nn.Parameter(torch.from_numpy(np.array([1])).cuda().float())
        
        #F4 Gaussian function 2
        self.a4 = nn.Parameter(torch.from_numpy(np.array([2])).cuda().float())
        self.c4 = nn.Parameter(torch.from_numpy(np.array([1])).cuda().float())
        
        self.n_batch = nn.BatchNorm1d(input_size, affine = False)
        
        
        
        """ set all consequent parameters [single layer] """
        self.fc1 = nn.Linear(input_size, ouput_size)
        self.fc2 = nn.Linear(input_size, ouput_size)

        
    def forward(self, x):
       
    
        x = self.n_batch(x)
        self.dropout = nn.Dropout(p = 0.2)
        
        # layer1
        # Rule 1
        miu1, tag = self.randomSelectMF(x)
        # Rule 2
        miu2, tag = self.randomSelectMF(x)
        
               
        # layer2 - firing strength of a rule
        w1 = self.mulElements(x.size(0), miu1, tag)
        w2 = self.mulElements(x.size(0), miu2, tag)
                  
                
        # layer3 - cal ratio
        w1_bar = w1 / (w1 + w2)
        w2_bar = w2 / (w1 + w2)
        
        # layer4 - one layer        
        out1 = w1_bar * self.fc1(x)       
        out2 = w2_bar * self.fc2(x)
        

        # layer5 weighted sum
        out = (out1 + out2)

        print ('out:', out)
        
        return out

    
    
    
"""ANFIS for selecting consequent parameters"""    
class ANFIS_consequent(nn.Module):
 
    def __init__(self, input_size = 9, ouput_size = 1, pre_group, layer_num):
        super(ANFIS, self).__init__()
        
        """ set all premise parameters, then fixing"""
        
        # precess premise group
        pre_list = []
        for pre_mf_param in pre_group:
            for i in range(len(pre_mf_param)):
                pre_list.append(pre_mf_param[i])
                
        
        """fixing premise"""
        #F1 generalized bell function 1
        self.a1 = nn.Parameter(torch.from_numpy(np.array([pre_list[0]])).cuda().float())
        self.b1 = nn.Parameter(torch.from_numpy(np.array([pre_list[1]])).cuda().float())
        self.c1 = nn.Parameter(torch.from_numpy(np.array([pre_list[2]])).cuda().float())
                    
        #F2  generalized bell function 2
        self.a2 = nn.Parameter(torch.from_numpy(np.array([pre_list[3]])).cuda().float())
        self.b2 = nn.Parameter(torch.from_numpy(np.array([pre_list[4]])).cuda().float())        
        self.c2 = nn.Parameter(torch.from_numpy(np.array([pre_list[5]])).cuda().float()) 
        
        #F3  Gaussian function 1
        self.a3 = nn.Parameter(torch.from_numpy(np.array([pre_list[6]])).cuda().float())
        self.c3 = nn.Parameter(torch.from_numpy(np.array([pre_list[7]])).cuda().float())
        
        #F4 Gaussian function 2
        self.a4 = nn.Parameter(torch.from_numpy(np.array([pre_list[8]])).cuda().float())
        self.c4 = nn.Parameter(torch.from_numpy(np.array([pre_list[9]])).cuda().float())
        
        self.n_batch = nn.BatchNorm1d(input_size, affine = False)
        
        
        
        """ adjusting consequent parameters """
        if layer_num==1:
                
            self.fc1 = nn.Linear(input_size, ouput_size)     
            self.fc2 = nn.Linear(input_size, ouput_size)
        
        if layer_num==3:

            self.fc1 = nn.Sequential(nn.Linear(input_size, 256), nn.Linear(256, 128), nn.Linear(128, 1)) 
            self.fc2 = nn.Sequential(nn.Linear(input_size, 256), nn.Linear(256, 128), nn.Linear(128, 1))

        if layer_num==6:

            self.fc1 = nn.Sequential(nn.Linear(input_size, 256), nn.Linear(256, 128), nn.Linear(128, 1))*2
            self.fc2 = nn.Sequential(nn.Linear(input_size, 256), nn.Linear(256, 128), nn.Linear(128, 1))*2
        
        if layer_num==9:
            self.fc1 = nn.Sequential(nn.Linear(input_size, 256), nn.Linear(256, 128), nn.Linear(128, 1))*3
            self.fc2 = nn.Sequential(nn.Linear(input_size, 256), nn.Linear(256, 128), nn.Linear(128, 1))*3    
            

        
    def forward(self, x):
       
    
        x = self.n_batch(x)
        self.dropout = nn.Dropout(p = 0.2)
        
        miu1, tag = self.randomSelectMF(x)
        miu2, tag = self.randomSelectMF(x)
        
               
        # layer2 - firing strength of a rule
        w1 = self.mulElements(x.size(0), miu1, tag)
        w2 = self.mulElements(x.size(0), miu2, tag)
                  
                
        # layer3 - cal ratio
        w1_bar = w1 / (w1 + w2)
        w2_bar = w2 / (w1 + w2)
        
        # layer4 - one layer        
        out1 = w1_bar * self.fc1(x)       
        out2 = w2_bar * self.fc2(x)
        

        # layer5 weighted sum
        out = (out1 + out2)

        print ('out:', out)
        
        return out
    
    
       
"""utility"""
# selecting different membership functions    
def randomSelectMF(self, x):
    
    rand_id = np.random.randint(0,4,1)
    if rand_id==0:
        miu = 1 / (1 + ((x - self.c1)**2/(self.a1)**2)**(self.b1))
        return miu, "B"
        
    if rand_id==1:       
        miu = 1 / (1+((x - self.c2)**2/(self.a2)**2)**(self.b2))
        return miu, "B"
    
    if rand_id==2:       
        miu = -((x - self.c3)**2 / self.a3**2)
        return miu, "G"
        
    if rand_id==3:       
        miu = -((x - self.c4)**2 / self.a4**2)
        return miu, "G"        
        
    
def mulElements(sample_nums, miu, tag):
    
    w = torch.zeros([sample_nums])
    w_tempB = 1
    w_tempG = 0
    
    if tag=="B":
        
        for i in range(miu.size(0)):
            for j in range(miu.size(1)):
                w_tempB = miu[i][j] * w_tempB

            w[i] = w_tempB
            w_tempB = 1
    
    if tag=="G":
        
        for i in range(miu.size(0)):
            for j in range(miu.size(1)):
                w_tempG = miu[i][j] + w_tempG

            w[i] = math.exp(w_tempG)
            w_tempG = 0        
        
    return w.cuda().float()

