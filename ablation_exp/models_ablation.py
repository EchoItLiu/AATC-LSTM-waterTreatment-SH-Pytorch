import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import math
from torch.nn.utils import weight_norm
import numpy as np


# daggerLSTM - multi-layer

class AALSTM(nn.Module):
    def __init__(self, in_features, trainSeq, testSeq):
        super(daggerTENLar, self).__init__()
        
        # init lstm layer
        self.lstm = nn.LSTMCell(in_features, 512)
        self.lstm2 = nn.LSTMCell(512, 256)
#         self.lstm3 = nn.LSTMCell(256, 128)
        
        self.dropout = nn.Dropout(p = 0.2)
        self.fc = nn.Linear(256, 1)
        
        # init BatchNormalization
        self.BN1d = nn.BatchNorm1d(in_features, affine = True)
        
        self.trainSeq = trainSeq
        self.testSeq = testSeq
        
    def forward(self, x, dagger_gt, is_train = False):
        outputs = []
        # init dagger factor
        alpha = 0
        
        if is_train == True:
            
            # BatchNormalization
            x_inver = torch.transpose(x,1,2)
            x_bn = self.BN1d(x_inver)
            x = torch.transpose(x_bn,1,2)
            
            # init hiddens and cell memories for all layers 
            h_0 = torch.zeros(x.size(0), 512).cuda().double()
            c_0 = torch.zeros(x.size(0), 512).cuda().double()
            
            h_1 = torch.zeros(x.size(0), 256).cuda().double()
            c_1 = torch.zeros(x.size(0), 256).cuda().double()
            
#             h_2 = torch.zeros(x.size(0), 128).cuda().double()
#             c_2 = torch.zeros(x.size(0), 128).cuda().double()                        
            
            # init (timestamp = 0)
            h_0, c_0 = self.lstm(x[:,0,:], (h_0, c_0))
            h_1, c_1 = self.lstm2(h_0, (h_1, c_1))
#             h_2, c_2 = self.lstm3(h_1, (h_2, c_2))
            
            out = self.fc(h_1)
            outputs.append(out)
        
            # timestamp ≥ 1
            for i in range(self.trainSeq - 1):

                beta = np.random.uniform(0,1,1)

                if alpha < beta:
                    # dagger: previous timestamp target → last dimension
                    input_pred = x[:,i+1,:-1]
#                     x[:,i+1,-1] = dagger_gt[:,i]  in-place                                 
                    input_train = torch.cat((input_pred, dagger_gt[:,i].view(-1,1)),1).squeeze() 
                    
                    h_0, c_0 = self.lstm(input_train, (h_0, c_0))
                    h_0 = self.dropout(h_0)
                    h_1, c_1 = self.lstm2(h_0, (h_1, c_1))
                    h_1 = self.dropout(h_1)
#                     h_2, c_2 = self.lstm3(h_1, (h_2, c_2))
#                     h_2 = self.dropout(h_2)
                    
                    out = self.fc(h_1)
#                     print ('out_size:', out.size())                    

                else:
                
                    # replace ouput from previous timestamp 
                    input_pred = x[:,i+1,:-1]
#                     x[:,i+1,-1] = out.squeeze()  in-place                     
                    input_train = torch.cat((input_pred, out.view(-1,1)), 1).squeeze()
    
                    h_0, c_0 = self.lstm(input_train, (h_0, c_0))
                    h_0 = self.dropout(h_0)
                    h_1, c_1 = self.lstm2(h_0, (h_1, c_1))
                    h_1 = self.dropout(h_1)
#                     h_2, c_2 = self.lstm3(h_1, (h_2, c_2))
#                     h_2 = self.dropout(h_2)                    
                    out = self.fc(h_1)

                alpha = alpha + 3e-2
                outputs.append(out)


        else:
            
            # BatchNormalization
            x_inver = torch.transpose(x,1,2)
            x_bn = self.BN1d(x_inver)
            x = torch.transpose(x_bn,1,2)           
            
            h_0 = torch.zeros(x.size(0), 512).cuda().double()
            c_0 = torch.zeros(x.size(0), 512).cuda().double()
            
            h_1 = torch.zeros(x.size(0), 256).cuda().double()
            c_1 = torch.zeros(x.size(0), 256).cuda().double()
            
#             h_2 = torch.zeros(x.size(0), 128).cuda().double()
#             c_2 = torch.zeros(x.size(0), 128).cuda().double()              
                        
            # init (timestamp = 0)
            h_0, c_0 = self.lstm(x[:,0,:], (h_0, c_0))
            h_1, c_1 = self.lstm2(h_0, (h_1, c_1))
#             h_2, c_2 = self.lstm3(h_1, (h_2, c_2))
                        
            out = self.fc(h_1)
            outputs.append(out)
            
            #  timestamp ≥ 1
            for i in range(self.testSeq - 1):
                input_pred = x[:,i+1,:-1]
#                 x[:,i+1,-1] = out.squeeze()  in-place                
                input_train = torch.cat((input_pred, out.view(-1,1)), 1).squeeze()
    
                h_0, c_0 = self.lstm(input_train, (h_0, c_0))
                h_0 = self.dropout(h_0)
                h_1, c_1 = self.lstm2(h_0, (h_1, c_1))
                h_1 = self.dropout(h_1)
#                 h_2, c_2 = self.lstm3(h_1, (h_2, c_2))
#                 h_2 = self.dropout(h_2)                    
                out = self.fc(h_1)
                outputs.append(out)
    
        outputs = torch.stack(outputs).squeeze()
        outputs = outputs.transpose(0,1)
                
        return outputs

        
# LSTM    
class LSTM(nn.Module):
    def __init__(self, in_features):
        super(lstmTENLar, self).__init__()
        
        self.lstm = nn.LSTMCell(in_features, 256)
        self.fc = nn.Linear(256, 1)
        
        
    def forward(self, x, Seq):
        outputs = []
        
        h_t = torch.zeros(x.size(0), 256).cuda().double()
        c_t = torch.zeros(x.size(0), 256).cuda().double()
        batch_size = x.size(0)
            
            
        # init (timestamp = 0)
        h_t, c_t = self.lstm(x[:,0,:], (h_t, c_t))
        out = self.fc(h_t)
        outputs.append(out)
        
        # timestamp ≥ 1
        for i in range(Seq - 1):

            input_pred = x[:,i+1,:-1]
            input_train = torch.cat((input_pred, out.view(batch_size,1)), 1).squeeze()                   
            h_t, c_t = self.lstm(input_train, (h_t, c_t))
            out = self.fc(h_t)
            outputs.append(out)   
    
        outputs = torch.stack(outputs).squeeze()
        outputs = outputs.transpose(0,1)
                
        return outputs    
        