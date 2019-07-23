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
        
# TCN    
class TCN(nn.Module):
    
    # warning print (in_channels)
    def __init__(self, in_channels = 9, out_channels = 256, kernel_size = 3, stride = 1, padding = 1):
        super(deepTENLar, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 512, kernel_size, stride, padding, bias = True)        
#         self.conv1 = weight_norm(nn.Conv1d(in_channels, 512, kernel_size, stride, padding, bias = True))

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p = 0.2)
        
        self.conv2 = nn.Conv1d(512, out_channels, kernel_size, stride, padding, bias = True)        
#         self.conv2 = weight_norm(nn.Conv1d(512, out_channels, kernel_size, stride, padding, bias = True))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p = 0.2)
        
        self.deepcnn = nn.Sequential(self.conv1, 
                                     self.relu1, 
                                     self.dropout1,
                                     self.conv2, 
                                     self.relu2, 
                                     self.dropout2)
        
        # FC
        self.linear = nn.Linear(128, 1, bias = True)

        
    def forward(self, x):
        y_deep = self.deepcnn(x)
#         print ('immediate_conv1d_out:', y_deep.size())
        # tranpose axis →  batch × sequence × channel
        y_deep_trans = torch.transpose(y_deep,1,2)        
        y = self.linear(y_deep_trans).squeeze()        
#         print ('origin_out:', y)
        return y
    
    
class ANFIS(nn.Module):
 
    def __init__(self, input_size = 9, ouput_size = 1):
        super(ANFIS, self).__init__()
        
        # all Membership Grade parameters  
        # registration
        # Rule 1 params  only number not a array
        self.a1 = nn.Parameter(torch.from_numpy(np.array([1])).cuda().double())
#         self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.zeros(input_size))
        self.c1 = nn.Parameter(torch.from_numpy(np.array([0])).cuda().double())
#         self.c1 = nn.Parameter(nn.init.xavier_normal_(torch.zeros(input_size)))
                    
        # Rule 2 params  
        self.a2 = nn.Parameter(torch.from_numpy(np.array([1])).cuda().double())
        self.c2 = nn.Parameter(torch.from_numpy(np.array([0])).cuda().double())   
        
        self.n_batch = nn.BatchNorm1d(1, affine = False)
        
          
        # Rule 1 fc:
        self.fc1 = nn.Linear(input_size, ouput_size)
#         self.fc1_a = nn.Linear(in, out)
        
        # Rule 2 fc:
        self.fc2 = nn.Linear(input_size, ouput_size)

        
    def forward(self, x):
        
        # layer1 - Fuzzification 
         # Rule 1
        
#         print ('anfis_c1:', self.c1)
#         print ('anfis_a1:', self.a1)
        miu1_bone = -((x - self.c1) / self.a1)**2
         # Rule 2
        miu2_bone = -((x - self.c2) / self.a2)**2
#         print ('miu2_size:', miu2.size())
#         print ('miu2:', miu2)        
         # Rule 1
          
         # Batch Normalization
           # Rule 1 BN
#         sum_miu1_ne = - torch.sum(miu1, 1)
        
        # channel is equivalent to degree of membership per sample
        # Batch Normalization
#         sum_miu1_bn = self.n_batch(sum_miu1_ne.view(-1, 1))
#         sum_miu1_bn = sum_miu1_ne.view(len_miu_ne, 1)
        
        
#         exp_sum_miu1_bn = torch.exp(sum_miu1_bn)       
#         print ('sum_miu1_bn:', sum_miu1_bn)
#         print ('sum_miu1_bn_size:', sum_miu1_bn.size())
        
#         print ('exp_sum_miu1_bn:', exp_sum_miu1_bn)
        
           # Rule 2 BN
#         sum_miu2_ne = - torch.sum(miu2, 1)
#         sum_miu2_bn = self.n_batch(sum_miu2_ne.view(-1, 1))
#         sum_miu2_bn = sum_miu2_ne.view(len_miu_ne, 1)
       
        # layer2 - Fuzzification      
        miu1 = torch.exp(miu1_bone)
#         print ('w1_size:', w1.size())
#         print ('w1:', w1)
         # Rule 2
        miu2 = torch.exp(miu2_bone)
#         print ('w2_size:', w2.size())
#         print ('w2:', w2)        
        
        # layer2 - firing strength of a rule
        w1 = mulElements(miu1)
        w2 = mulElements(miu2)
     
                
        # layer3 - cal ratio
        w1_bar = w1 / (w1 + w2)
        w2_bar = w2 / (w1 + w2)
   
        
        # layer4 + layer5: weights and outputs
         # Rule 1 weighted FC
        out1 = w1_bar * self.fc1(x)
#         print ('out1_size:', out1.size())
#         print ('out1:', out1)        
        out2 = w2_bar * self.fc2(x)
        
         # Rule 2 weighted FC
#         print ('out2_size:', out2.size())
#         print ('out2:', out2)          
        
        # layer5 weighted sum
        out = out1 + out2
#         print ('anfis_out:', out)
#         print ('anfis_out_size:', out.size())

        print (out)
        
        return out
    

    
# daggerLSTM
# class daggerTENLar(nn.Module):
#     def __init__(self, in_features, trainSeq, testSeq):
#         super(daggerTENLar, self).__init__()
        
#         self.lstm = nn.LSTMCell(in_features, 256)
#         self.fc = nn.Linear(256, 1)
        
#         # init BatchNormalization
#         self.BN1d = nn.BatchNorm1d(in_features, affine = False)
        
#         self.trainSeq = trainSeq
#         self.testSeq = testSeq
        
#     def forward(self, x, dagger_gt, is_train = False):
#         outputs = []
#         # init dagger factor
#         alpha = 0
        
#         if is_train == True:
            
#             # BatchNormalization
#             x_inver = torch.transpose(x,1,2)
#             x_bn = self.BN1d(x_inver)
#             x = torch.transpose(x_bn,1,2)
            
#             h_t = torch.zeros(x.size(0), 256).cuda().double()
#             c_t = torch.zeros(x.size(0), 256).cuda().double()
#             batch_size = x.size(0)
            
            
#             # init (timestamp = 0)
#             h_t, c_t = self.lstm(x[:,0,:], (h_t, c_t))
#             out = self.fc(h_t)
#             outputs.append(out)
        
#             # timestamp ≥ 1
#             for i in range(self.trainSeq - 1):

# #                 beta = np.random.uniform(0,1,1)

#                 if alpha < beta:
#                     # dagger: previous timestamp target → last dimension
#                     input_pred = x[:,i+1,:-1]
# #                     x[:,i+1,-1] = dagger_gt[:,i]  in-place                    
    
            
#                     input_train = torch.cat((input_pred, dagger_gt[:,i].view(batch_size,1)),1).squeeze() 
                    
#                     h_t, c_t = self.lstm(input_train, (h_t, c_t))
#                     out = self.fc(h_t)
# #                     print ('out_size:', out.size())                    

#                 else:
#                     # replace ouput from previous timestamp 
#                     input_pred = x[:,i+1,:-1]
# #                     x[:,i+1,-1] = out.squeeze()  in-place                     
#                     input_train = torch.cat((input_pred, out.view(batch_size,1)), 1).squeeze()
#                     h_t, c_t = self.lstm(input_train, (h_t, c_t))
#                     out = self.fc(h_t)


#                 alpha = alpha + 1e-1
#                 outputs.append(out)
    
    
    
# class daggerTENLar(nn.Module):
#     def __init__(self, in_features, trainSeq, testSeq):
#         super(daggerTENLar, self).__init__()
    
#     # init LSTM
#     self.lstm = nn.LSTM(input_size = in_features, hidden_size = 128, num_layers = 3, bias = True, batch_first = True, dropout = 0, bidirectional = False)
    
#         self.fc = nn.Linear(128, 1)
        
#         # init BatchNormalization
#         self.BN1d = nn.BatchNorm1d(in_features, affine = False)
        
#         self.trainSeq = trainSeq
#         self.testSeq = testSeq
        
#     def forward(self, x, dagger_gt, is_train = False):
#         outputs = []
#         # init dagger factor
#         alpha = 0
        
#         if is_train == True:
            
#             # BatchNormalization
#             x_inver = torch.transpose(x,1,2)
#             # batch × seq × channel
#             x_bn = self.BN1d(x_inver)
#             # seq × batch × channel
#             x = torch.transpose(x,0,1)
            
            
#             h_0 = torch.zeros(3 * 1, x.size(1), 128).cuda().double()
#             c_0 = torch.zeros(3 * 1, x.size(1), 128).cuda().double()
                      
#             # init (timestamp = 0)
#             h_t, c_t = self.lstm(x, (h_0, c_0))
#             out = self.fc(h_t)
        
#         else:
#             h_0 = torch.zeros(3 * 1, x.size(1), 128).cuda().double()
#             c_0 = torch.zeros(3 * 1, x.size(1), 128).cuda().double() 
            
            
#             # init (timestamp = 0)
#             h_t, c_t = h_t, c_t = self.lstm(x, (h_0, c_0))
#             out = self.fc(h_t)
            
                
#         return outputs


def mulElements(miu):
    w = torch.zeros([8000])
    w_temp = 1
    for i in range(miu.size(0)):
        for j in range(miu.size(1)):
            w_temp = miu[i][j] * w_temp 
                       
        w[i] = w_temp
        
    return w