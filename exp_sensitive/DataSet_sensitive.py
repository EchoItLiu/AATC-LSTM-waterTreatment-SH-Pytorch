from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os.path as osp
import numpy as np
import xlrd
import config_ablation 
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN 
from sklearn.decomposition import PCA

class DataSet:
    def __init__(self, excel_files):
        cache_filename = 'data/raw_data.csv'
        if osp.exists(cache_filename):
            self.raw_data = pd.read_csv(cache_filename)
#             Convert argument to datetime(to_datetime)
#             year-month-day
            self.raw_data['Time'] = pd.to_datetime(self.raw_data['Time'])
        else:
            frame_list = []
            for file_name in excel_files:
                book = xlrd.open_workbook(osp.join('data', file_name))
                for sheet in book.sheets():
                    try:
                        df = pd.read_excel(osp.join('data', file_name), sheet.name, header=None, names=config_ablation.attributes)
                    except:
                        continue
                    df.drop([0,1], inplace=True)
                    frame_list.append(df)
            self.raw_data = pd.concat(frame_list)
            self.raw_data.sort_values(by='Time')
            # delete NAN for different axis
            self.raw_data.dropna(axis=0, how='all')
            self.raw_data.dropna(axis=1, how='all')
            # export single csv_file including all data above(time-series has interval)
            self.raw_data.to_csv(cache_filename) 

    def select_time(self, time_intervals, frame):
        for i,[start_time, end_time] in enumerate(time_intervals):
#             ('year-month-day xx:xx:xx')
            start_time = pd.Timestamp(start_time)
            end_time = pd.Timestamp(end_time) 
            if i==0:
                index = (frame['Time'] >= start_time) & (frame['Time'] < end_time)
            else:
                index |= (frame['Time'] >= start_time) & (frame['Time'] < end_time)
        return frame[index]
    
    # reshape to 3D Tensor
    def feature2series(self, x, lag_time):
        n = x.shape[0]
        new_x = x.reshape([n, lag_time, -1])
        return new_x
    
    def select_attributes(self, attributes, frame):
        new_frame = pd.DataFrame(columns=attributes)
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
            #  all the other features have average
            else:
                new_frame[attr] = frame[[s for s in frame.columns if attr in s]].mean(1)
        return new_frame     
      
    def attr2source(self, time, normalize, clustering, lag_time, data_type='train'):
        self.lag_time = lag_time
        frame = self.select_time(time, self.raw_data)
        # ***including time dimensions not including time
        features = self.select_attributes(config_ablation.features, frame)
        # including time dimension
        labels = self.select_attributes(config_ablation.labels, frame)
        x = []
        index = True
        for i in range(lag_time):
            index &= (labels['OD'].shift(-i)>20) & (labels['OD'].shift(-i)<60) & (np.abs(labels['OD'].shift(-i-1) -labels['OD'].shift(-i))<10) & (labels['Time'].shift(-i-1)-labels['Time'].shift(-i)==timedelta(minutes=5))
            # have additional time dimensions and OD from previous lags to list
            x.extend([features.shift(-i-1), labels['OD'].shift(-i)])
#       list → pandas   
        x = pd.concat(x, 1)
        # labels times
        y = labels['OD'].shift(-lag_time)
        
#         print ('Time:',labels['Time'])
        
        # drop optimal dosage >=60 or <=20, diff>=15, and time-diff!=5min
        # drop nan, inf
        # △△△ time-series broken down
        x.replace([np.inf, -np.inf], np.nan)
        y.replace([np.inf, -np.inf], np.nan)
        index &= (y>20)&(y<60)&(np.sum(x==0,1)==0)&(pd.isnull(x).sum(1)==0)&(pd.isnull(y)==0)
        x = x[index]
        y = y[index]
        
        # additional 19 dimensions(△△△)
#         print ('original_df_X:', x)
        
#         print ('original_df_Y:', y)
        # transform to numpy
    
        # additional 82 dimensions
        x = x.values
        y = y.values
        
#         print ('X:',x.shape)
#         print ('Y:',y.shape)
        
       # save all data
        if data_type == 'all':
            self.all_x, self.all_y = x,y 
#             print ('self_all_x_shape:',self.all_x.shape)
#             print ('self_all_y_shape:',self.all_y.shape)

#             print ('self_all_x:',self.all_x) 
#             print ('self_all_y:',self.all_y) 
        
        
        if data_type =='time-series':
            
#             print ('--------------------')
            # sequence length 
            sample_len = 10000
            # set random initial point for sequence
            sample_ip= np.random.randint(0,(13000 - sample_len + 1) )
            
#             print ('sample_ip:',sample_ip)
#             print ('x_shape:', x.shape)
#             print ('y_shape:', y.shape)
            
#             print ('x_limit:',x.shape)
#             print ('y_limit:',y.shape)
        
            self.ts_x = x[sample_ip:(sample_ip + sample_len),:]
            self.ts_y = y[sample_ip:(sample_ip + sample_len)]
            
#             print ('self_ts_x:',self.ts_x) 
#             print ('self_ts_y:',self.ts_y) 
            
#             print  ('self_ts_x_shape:',self.ts_x.shape)
#             print  ('self_ts_y_shape:',self.ts_y.shape)

#             print ('--------------------')

           
            # normalize             
            if normalize == True:
                self.normalizer = StandardScaler()
                self.ts_x = self.normalizer.fit_transform(self.ts_x )
     
    
            
        
        if data_type == 'train':
            # clustering
            if clustering == True:
                kmeans = KMeans(n_clusters=1000)
                cluster_labels = kmeans.fit_predict(x)
                idx = []
                for label in np.unique(cluster_labels):
                    iidx = np.where(label==cluster_labels)[0]
                    idx.append(iidx[np.random.randint(0,iidx.shape[0])])
                x = x[idx]
                y = y[idx]
            # random
            else:
                idx = np.random.randint(0, x.shape[0], 1000)
                x = x[idx]
                y = y[idx]
 
            # normalize             
            if normalize == True:
                self.normalizer = StandardScaler()
                x = self.normalizer.fit_transform(x)

            
        else:
            idx = np.random.randint(0, x.shape[0], 1000)
            x = x[idx]
            y = y[idx]
            
            # normalize             
            if normalize == True:
                self.normalizer = StandardScaler()
                x = self.normalizer.fit_transform(x)
                        
        
        if data_type == 'train':
            self.train_x, self.train_y = x,y
            
#             print ('self.train_x_shape:' , self.train_x.shape)
#             print ('self.train_y_shape:', self.train_y.shape)
#             print ('self.train_x:' , self.train_x)
#             print ('self.train_y:' , self.train_y)
            

            
        else:
            self.test_x, self.test_y = x,y


        

        

       
        
       

       

        
    