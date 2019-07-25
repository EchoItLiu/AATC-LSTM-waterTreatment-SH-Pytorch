from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import os.path as osp
import numpy as np
import xlrd
import config
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN 
from sklearn.decomposition import PCA

class DataSet:
    def __init__(self, excel_files):
        cache_filename = 'data/raw_data.csv'
        
        # reading if .csv exists
        if osp.exists(cache_filename):
            self.raw_data = pd.read_csv(cache_filename)
            # format timestamp in column ‘Time’
            self.raw_data['Time'] = pd.to_datetime(self.raw_data['Time'])
        
        # concat data from 3 files ??? sequence error ???
        else:
            frame_list = []
            # 
            for file_name in excel_files:
                # all sheets list
                book = xlrd.open_workbook(osp.join('data', file_name))
                # for every sheet 
                for sheet in book.sheets():
                    try:
                        # read data to dataframe via sheet name(0/1/2...)
                        df = pd.read_excel(osp.join('data', file_name), sheet.name, header=None, names=config.attributes)
                    except:
                        continue
                    # delete frist 2 rows in every dataframe
                    df.drop([0,1], inplace = True)
                    # list append
                    frame_list.append(df)
            # concat pandas objects along a particular axis
            self.raw_data = pd.concat(frame_list)
            # Sort by the values along either axis (Times)
            self.raw_data.sort_values(by='Time')
            # remove NaN, If all values are NA, drop that row or column
            self.raw_data.dropna(axis=0, how='all')
            self.raw_data.dropna(axis=1, how='all')
            # save .csv
            self.raw_data.to_csv(cache_filename) 
            

    # get indice of timestamp       
    def select_time(self, time_intervals, frame):
        for i,[start_time, end_time] in enumerate(time_intervals):
            # set start - end timestamp(config-experiments-seasons-times)
            start_time = pd.Timestamp(start_time)
            end_time = pd.Timestamp(end_time) 
                       
            if i==0:
                index = (frame['Time'] >= start_time) & (frame['Time'] < end_time)
            else:
                index |= (frame['Time'] >= start_time) & (frame['Time'] < end_time)
        return frame[index]
    
    #  
    def feature2series(self, x, lag_time):
        n = x.shape[0]
        new_x = x.reshape([n, lag_time, -1])
        return new_x
    
    # 
    def select_attributes(self, attributes, frame):
        
        new_frame = pd.DataFrame(columns = attributes)
        for attr in attributes:
            if attr == 'OD':
                new_frame[attr] = 0.1527 * 1e3 * frame[['Al Dosage-1', 'Al Dosage-2', 'Al Dosage-3', 'Al Dosage-4']].sum(1)/ (frame[['Enter Water-1', 'Enter Water-2', 'Enter Water-3', 'Enter Water-4']].sum(1) + 1)
            elif attr == 'SW-TB1':
                new_frame[attr] = frame[['SW-TB-1', 'SW-TB-2', 'SW-TB-3', 'SW-TB-4']].mean(1)
            elif attr == 'SW-TB2':
                new_frame[attr] = frame[['PC-TB-1', 'PC-TB-2']].mean(1)
            elif attr.split('-')[0] == 'RW':
                new_frame[attr] = frame[attr]
            elif attr == 'Time':
                new_frame[attr] = frame[attr]
            else:
                new_frame[attr] = frame[[s for s in frame.columns if attr in s]].mean(1)
        return new_frame     
      
    def attr2source(self, time, normalize, clustering, lag_time, data_type='train'):
        self.lag_time = lag_time
        frame = self.select_time(time, self.raw_data)
        features = self.select_attributes(config.features, frame)
        labels = self.select_attributes(config.labels, frame)
        x = []
        index = True
        
        
        """9 the most important features +  time-lagged water features"""
        for i in range(lag_time):
            index &= (labels['OD'].shift(-i)>20) & (labels['OD'].shift(-i)<60) & (np.abs(labels['OD'].shift(-i-1) -labels['OD'].shift(-i))<10) & (labels['Time'].shift(-i-1)-labels['Time'].shift(-i)==timedelta(minutes=5))
            x.extend([features.shift(-i-1), labels['OD'].shift(-i)])
        x = pd.concat(x, 1)
        y = labels['OD'].shift(-lag_time)        
        x.replace([np.inf, -np.inf], np.nan)
        y.replace([np.inf, -np.inf], np.nan)
        index &= (y>20)&(y<60)&(np.sum(x==0,1)==0)&(pd.isnull(x).sum(1)==0)&(pd.isnull(y)==0)
        x = x[index]
        y = y[index]
        
        x = x.values
        y = y.values
        
        
        # get shape of origin samples 
        xShape = x.shape
        yShape = y.shape[0]
        
        print ('Origin_X_Shape:', x.shape)
#         print ('origin_y_shape:', y.shape)
        
        ## return all time-series data   
                       
            # Normalize between features             
        if normalize == True:
            self.normalizer = StandardScaler()
            x = self.normalizer.fit_transform(x)
                 
            # clustering
        if clustering == True:
            kmeans = KMeans(n_clusters=1000)
            cluster_labels = kmeans.fit_predict(x)
            idx = []
            for label in np.unique(cluster_labels):
                # idx pos in same label
                iidx = np.where(label==cluster_labels)[0]
                idx.append(iidx[np.random.randint(0,iidx.shape[0])])
            
                
            # remove samples in same labels
            x = x[idx]
            y = y[idx]
            
             
        ## return all time-series data
        if data_type =='time-series':
            
            # set multi-sample length            
            sample_len = 15000
            seq_len_nn = 3000
            
            
            sample_ip= np.random.randint(0,(16000 - sample_len + 1) )
            
            ## Neural Network
            self.nn_trainX = x[sample_ip:(sample_ip + seq_len_nn),:] 
            self.nn_trainY = y[sample_ip:(sample_ip + seq_len_nn)]
            
                                    
            
        if data_type =='ts-train':
            sample_len = 1000
            sample_ip= np.random.randint(0,(13000 - sample_len + 1) )                    
            self.x_train = x[sample_ip:(sample_ip + sample_len),:]
            self.y_train = y[sample_ip:(sample_ip + sample_len)]
#             print ('ts-train_shape:', self.x_train.shape)
            
            
        if data_type =='ts-test':
            seq_len_ml = 50
#             seq_len_ml = 10    
            seq_len_nn = 1000
#             seq_len_nn = 10
        
            x_channel = x.shape[1]
            
            sample_ip= np.random.randint(0,(6000 - sample_len + 1) )
            
            # test for any other baseline models
            self.x_test = x[sample_ip:(sample_ip + sample_len),:]
            self.y_test = y[sample_ip:(sample_ip + sample_len)] 
            
                 
            # ML models ndarray
            self.ml_x_test = self.x_test.reshape(-1, seq_len_ml, x_channel)
            self.ml_y_test = self.y_test.reshape(-1, seq_len_ml)
            
            
            ## Neural Network
            self.nn_testX = x[sample_ip:(sample_ip + seq_len_nn),:] 
            self.nn_testY = y[sample_ip:(sample_ip + seq_len_nn)]  
            
        if data_type =='anfis':
            
            sample_len_train = 3000
            sample_len_test = 1000
            
            feature_size = xShape[1]
            sample_ip_train = np.random.randint(0,(13000 - sample_len_train + 1) )  
            sample_ip_test = np.random.randint(0,(13000 - sample_len_test + 1) )  
            
            
            # train
            x_anfis_cpu_train = x[sample_ip_train:(sample_ip_train + sample_len_train)]
            y_anfis_cpu_train = y[sample_ip_train:(sample_ip_train + sample_len_train)]         
            self.x_anfis_train = torch.from_numpy(x_anfis_cpu_train).cuda()
            self.y_anfis_train = torch.from_numpy(y_anfis_cpu_train).cuda()
            
            # test
            x_anfis_cpu_test = x[sample_ip_test:(sample_ip_test + sample_len_test)]
            y_anfis_cpu_test = y[sample_ip_test:(sample_ip_test + sample_len_test)]         
            self.x_anfis_test = torch.from_numpy(x_anfis_cpu_test).cuda()
            self.y_anfis_test = torch.from_numpy(y_anfis_cpu_test).cuda()            
            
            
            print ('anfis_traindata_size:', self.x_anfis_train.size())
            
            
       ## return all data
        if data_type == 'all':
            self.all_x, self.all_y = x,y 
            

   