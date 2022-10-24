from ast import Str
from tokenize import String
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from skimage.util import view_as_windows
from utils import shift
# from fireTS.core.

class DataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(DataFrame, self).__init__(*args, **kwargs)

        self.scaler=None

    @property
    def _constructor(self):
        return type(self)

    def splitXY(self, X_features: List,
                        Y_features: List):
        '''Method for splitting a dataset in X and Y datasets

        :param List X_features: X features 
        :param List Y_features: Y features  
        
        :returns: X: train dataset
                    Y: Y train dataset

        :rtype: pextpdDataFrame
        '''

        X=self.get(X_features)
        Y=self.get(Y_features)

        return X, Y

    def splitTrainTest(self, *args, **kwargs):
        '''Method for splitting a dataset in X and Y datasets

        :param *arrays
        :param test_size    
        :param train_size 
        :param random_state
        :param shuffle 
        :param stratify 

        :returns: train_df: training dataframe
                    test_df: test dataframe 

        
        :rtype: extpandas.DataFrame
        '''
        train_df, test_df = train_test_split(self, *args, **kwargs)

        return train_df, test_df

    def Transform(self, transformation: String=None,
                        scaler=None,
                        *args, 
                        **kwargs):
        '''Method for transforming a dataset

        :param String transformation: transfomation 
        :param sklearn.preprovcessing.scaler scaler: scaler  

        :returns: scaler
       
        :rtype: sklearn.preprocessing.scaler
        '''   
        if scaler==None:
            if transformation=='StandardScaler': 
                scaler=preprocessing.StandardScaler(*args, **kwargs)
            
            elif transformation=='Normalizer':
                scaler=preprocessing.Normalizer(*args, **kwargs)
            
            elif transformation=='MinMaxScaler':
                scaler=preprocessing.MinMaxScaler(*args, **kwargs)  
            
            elif transformation=='RobustScaler':
                scaler=preprocessing.RobustScaler(*args, **kwargs)
            
            elif transformation=='TfidfVectorizer':
                scaler=text.TfidfVectorizer(*args, **kwargs)    

            scaler.fit(self.values)
        
        self.__init__(data=scaler.transform(self.values), columns=self.columns)
        self.scaler=scaler

        return None

    def InverseTransform(self,
                             *args, **kwargs):
        '''Method for inverse transforming a dataset
        '''
        self.__init__(self.scaler.inverse_transform(self.values), columns=self.columns)

        return None

    def getCorrMatrix(self, *args, **kwargs):
        '''Method for inverse transforming a dataset

        :returns: CorrMatrix
       
        :rtype: extpandas.DataFrame
        '''
        plt.figure()
        CorrMatrix=self.corr(*args, **kwargs)
        sn.heatmap(CorrMatrix, annot=True)
        plt.show()

        return CorrMatrix

    def ViewAsWindows(self,
                        window_length: int):
        '''Method for preprocessing timseries data 

        :param int window length: length (in samples) of the window
        
        :returns: data_as_windows_3D

        :rtype: numpy.ndarray
        '''

        data_as_windows_3D=np.squeeze(view_as_windows(self.values, (window_length, self.values.shape[1])))
        
        #shape=data_as_windows_3D.shape
        #data_as_windows=np.reshape(data_as_windows_3D, (shape[0]*shape[1], shape[2]))
        #self.__init__(data_as_windows, columns=self.columns)
        
        return data_as_windows_3D

    def NARX(self, ex_order: List = None,
                    ex_delay: List = None,
                    auto_order: int = None,
                    pred_step: int = None):
        '''Method for processing the data in a NARX fashion.
            y(t+k)=f(y(t), ..., y(t-p+1), 
                    x1(t-d1), ..., x1(t-d1-q1+1), ...
                    xm(t-dm), ..., xm(t-dm-qm+1), e(t))
            Where:
                q (ex_order): exogenous order 
                d (ex_delay): exogenous delay 
                p (auto_order): auto order
                k (pred_step): prediction step
        '''
        # shift dataframe 
        for d, column in enumerate(self.columns):
            #self[column]=self.columns.shift(ex_delay[d])
            
            self[column]=shift(self[column], ex_delay[d])

            return None
        